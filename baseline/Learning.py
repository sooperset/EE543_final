from pathlib import Path
import pandas as pd
import torch
from apex import amp
from tqdm import tqdm
from scripts.utils import empty_cuda_cache, reduce_tensor, one_hot_embedding
from scripts.tb_helper import simple_result, normalize_tensor2tensor, normalize_tensor2numpy
import heapq
import random
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import pdb

class Learning():
    def __init__(self,
                 distrib_config,
                 optimizer,
                 loss_fn,
                 device,
                 n_epoches,
                 scheduler,
                 accumulation_step,
                 early_stopping,
                 logger,
                 tb_logger,
                 best_checkpoint_folder,
                 checkpoints_history_folder,
                 checkpoints_topk,
                 calculation_name
        ):
        self.distrib_config = distrib_config
        self.logger = logger
        self.tb_logger = tb_logger
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.epoch = 0
        self.n_epoches = n_epoches
        self.scheduler = scheduler
        self.accumulation_step = accumulation_step
        self.early_stopping = early_stopping
        self.calculation_name = calculation_name
        self.best_checkpoint_path = Path(
            best_checkpoint_folder,
            '{}.pth'.format(self.calculation_name)
        )
        self.checkpoints_history_folder = checkpoints_history_folder
        self.score_heap = [(0., Path('nothing'))]
        self.summary_file = Path(self.checkpoints_history_folder, 'summary.csv')
        if self.summary_file.is_file():
            self.best_score = pd.read_csv(self.summary_file).best_metric.max()
            if self.distrib_config['LOCAL_RANK'] == 0:
                logger.info('Pretrained best score is {:.5}'.format(self.best_score))
        else:
            self.best_score = 0
        self.best_score = 0
        self.best_epoch = -1
        self.checkpoints_topk = checkpoints_topk
        # empty_cuda_cache()

    def train_epoch(self, model, loader):
        tqdm_loader = tqdm(loader)
        current_loss_mean = 0.
        for idx, batch in enumerate(tqdm_loader):
            if (idx + 1) % self.accumulation_step == 0:
                self.optimizer.zero_grad()

            loss = self.batch_train(model, batch[0], batch[1])
            current_loss_mean = (current_loss_mean * idx + loss.item()) / (idx + 1)

            if (idx + 1) % self.accumulation_step == 0:
                self.optimizer.step()

            tqdm_loader.set_description(f'loss: {current_loss_mean:.4f} lr: {self.optimizer.param_groups[0]["lr"]:.6f}')
        # empty_cuda_cache()
        return current_loss_mean

    def batch_train(self, model, batch_imgs, batch_labels):
        batch_imgs = batch_imgs.to(device=self.device, non_blocking=True)
        batch_labels = batch_labels.to(device=self.device, non_blocking=True, dtype=torch.int64)
        batch_pred = model(batch_imgs)
        loss = self.loss_fn(batch_pred, batch_labels) / self.accumulation_step

        # loss.backward()
        with amp.scale_loss(loss, self.optimizer, loss_id=0) as scaled_loss:
            scaled_loss.backward()
        return loss

    def valid_epoch(self, model, loader, local_metric_fn):
        tqdm_loader = tqdm(loader)
        current_score_mean = torch.tensor(0.).cuda()
        eval_list = []
        for idx, batch in enumerate(tqdm_loader):
            with torch.no_grad():
                batch_labels = batch[1].to(dtype=torch.int64)
                batch_pred = self.batch_valid(model, batch[0])
                # batch_labels = one_hot_embedding(batch_labels, 2).permute(0,3,1,2)[:,1,...]
                eval_list.append((batch_pred, batch_labels))
                score = local_metric_fn(batch_pred, batch_labels)
                current_score_mean = (current_score_mean * idx + score) / (idx + 1)

                tqdm_loader.set_description(f'score: {current_score_mean:.5f}')

                if idx == 0:
                    # self.tb_logger.add_image('label', simple_result(batch_labels[0], (batch_pred[0] > 0.5)), self.epoch)
                    def make_figure(img):
                        fig, ax = plt.subplots(1)
                        fig.set_figwidth(10)
                        fig.set_figheight(10)
                        ax.axis('off')
                        ax.imshow(normalize_tensor2numpy(img))
                        plt.tight_layout()
                        return fig
                    # self.tb_logger.add_figure('image', make_figure(batch_imgs[0].cpu()), self.epoch)

        if self.distrib_config['DISTRIBUTED']:
            current_score_mean = reduce_tensor(current_score_mean, self.distrib_config['WORLD_SIZE'])
        # empty_cuda_cache()
        return eval_list, current_score_mean.item()

    def batch_valid(self, model, batch_imgs):
        batch_imgs = batch_imgs.to(device=self.device, non_blocking=True)
        batch_pred = model(batch_imgs)
        return batch_pred.cpu()

    def process_summary(self, eval_list, global_metric_fn):
        self.logger.info('{} epoch: \t start searching thresholds....'.format(self.epoch))
        selected_score, thr = global_metric_fn(eval_list)

        epoch_summary = pd.DataFrame(
            data=[[self.epoch, selected_score, thr]],
            columns=['epoch', 'best_metric', 'best_thr']
        )

        if self.distrib_config['LOCAL_RANK'] == 0:
            self.logger.info(f'Epoch {self.epoch}: \t Calculated score: {selected_score:.6f}, thr: {thr}')
            self.tb_logger.add_scalar('Valid/score', selected_score, self.epoch)

            if not self.summary_file.is_file():
                epoch_summary.to_csv(self.summary_file, index=False)
            else:
                summary = pd.read_csv(self.summary_file)
                summary = summary.append(epoch_summary)
                summary.to_csv(self.summary_file, index=False)

        return selected_score

    @staticmethod
    def get_state_dict(model):
        if type(model) == torch.nn.DataParallel:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        return state_dict

    def post_processing(self, score, model):
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = self.epoch
            if self.distrib_config['LOCAL_RANK'] == 0:
                torch.save(self.get_state_dict(model), self.best_checkpoint_path)
                self.logger.info('best model: {} epoch - {:.5}'.format(self.epoch, score))

        if self.distrib_config['LOCAL_RANK'] == 0:
            if self.score_heap[0][0] < score:
                checkpoints_history_path = Path(
                    self.checkpoints_history_folder,
                    '{}_epoch{}.pth'.format(self.calculation_name, self.epoch)
                )
                torch.save(self.get_state_dict(model), checkpoints_history_path)
                heapq.heappush(self.score_heap, (score, checkpoints_history_path))
                if len(self.score_heap) > self.checkpoints_topk:
                    _, removing_checkpoint_path = heapq.heappop(self.score_heap)
                    removing_checkpoint_path.exists() and removing_checkpoint_path.unlink()
                    self.logger.info('Removed checkpoint is {}'.format(removing_checkpoint_path))

        if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            self.scheduler.step(score)
        else:
            self.scheduler.step()

    def inference(self, model, loader):
        empty_cuda_cache()
        state_dict = torch.load(self.best_checkpoint_path)
        model.load_state_dict(state_dict)
        tqdm_loader = tqdm(loader)
        for idx, batch in enumerate(tqdm_loader):
            with torch.no_grad():
                batch_imgs = batch[0].to(device=self.device, non_blocking=True)
                batch_pred = model(batch_imgs).cpu()
                batch_pred = F.softmax(batch_pred, dim=1)[:, 1, ...]
                for pred_idx, pred in enumerate(batch_pred):
                    save_image(pred, self.checkpoints_history_folder / f'{pred_idx}.tif')

    def run_train(self, model, train_dataloader, valid_dataloader, local_metric_fn, global_metric_fn):
        model.to(self.device)
        model, self.optimizer = amp.initialize(model, self.optimizer, opt_level='O1')
        for self.epoch in range(self.n_epoches):
            if self.distrib_config['LOCAL_RANK'] == 0:
                self.logger.info(f'Epoch {self.epoch}: \t start training....')
            model.train()
            train_loss_mean = self.train_epoch(model, train_dataloader)
            if self.distrib_config['LOCAL_RANK'] == 0:
                self.logger.info(f'Epoch {self.epoch}: \t Calculated train loss: {train_loss_mean:.5f}')
                self.tb_logger.add_scalar('Train/Loss', train_loss_mean)

            if self.distrib_config['LOCAL_RANK'] == 0:
                self.logger.info(f'Epoch {self.epoch}: \t start validation....')
            model.eval()
            eval_list, valid_score_mean = self.valid_epoch(model, valid_dataloader, local_metric_fn)
            selected_score = self.process_summary(eval_list, global_metric_fn)

            self.post_processing(selected_score, model)

            if self.epoch - self.best_epoch > self.early_stopping:
                if self.distrib_config['LOCAL_RANK'] == 0:
                    self.logger.info('EARLY STOPPING')
                break

        if self.distrib_config['LOCAL_RANK'] == 0:
            self.tb_logger.close()

        self.inference(model, valid_dataloader)
        empty_cuda_cache()
        return self.best_epoch, self.best_score


