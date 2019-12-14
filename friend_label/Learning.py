from pathlib import Path
import numpy as np
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
                 optimizer1,
                 optimizer2,
                 loss_fn,
                 evaluator1,
                 evaluator2,
                 device1,
                 device2,
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
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.loss_fn = loss_fn
        self.evaluator1 = evaluator1
        self.evaluator2 = evaluator2
        self.device1 = device1
        self.device2 = device2
        self.epoch = 0
        self.n_epoches = n_epoches
        self.scheduler = scheduler
        self.accumulation_step = accumulation_step
        self.early_stopping = early_stopping
        self.calculation_name = calculation_name
        self.best_checkpoint_path1 = Path(
            best_checkpoint_folder,
            'model1_{}.pth'.format(self.calculation_name)
        )
        self.best_checkpoint_path2 = Path(
            best_checkpoint_folder,
            'model2_{}.pth'.format(self.calculation_name)
        )
        self.checkpoints_history_folder = checkpoints_history_folder
        self.score_heap = [(0., Path('nothing'))]
        self.summary_file = Path(self.checkpoints_history_folder, 'summary.csv')
        if self.summary_file.is_file():
            self.best_score = pd.read_csv(self.summary_file).MIoU.max()
            if self.distrib_config['LOCAL_RANK'] == 0:
                logger.info('Pretrained best score is {:.5}'.format(self.best_score))
        else:
            self.best_score = 0
        self.best_score = 0
        self.best_epoch = -1
        self.checkpoints_topk = checkpoints_topk
        # empty_cuda_cache()

    def correct_label(self, batch_pred1, batch_pred2, batch_labels):
        batch_size, out_channels, H, W = batch_pred1.size()
        batch_labels = batch_labels.long()

        with torch.no_grad():
            logp1 = F.log_softmax(batch_pred1, dim=1)
            logp2 = F.log_softmax(batch_pred2, dim=1)
            logp2 = logp2.to(self.device1)

            e = torch.eq(logp1.argmax(1), logp2.argmax(1))
            e = torch.where(batch_labels == 255, e, torch.zeros_like(e).bool())

            corrected = torch.where(e, logp1.argmax(1), batch_labels)

        return corrected.data


    def train_epoch(self, model1, model2, loader):
        tqdm_loader = tqdm(loader)
        current_loss1_mean = 0.
        current_loss2_mean = 0.
        for idx, batch in enumerate(tqdm_loader):
            image, target = batch['image'], batch['label']
            loss1, loss2 = self.batch_train(model1, model2, image, target, idx)
            current_loss1_mean = (current_loss1_mean * idx + loss1.item()) / (idx + 1)
            current_loss2_mean = (current_loss2_mean * idx + loss2.item()) / (idx + 1)

            tqdm_loader.set_description(f'loss1: {current_loss1_mean:.4f} lr: {self.optimizer1.param_groups[0]["lr"]:.6f}'
                                        f'loss2: {current_loss2_mean:.4f}')
        # empty_cuda_cache()
        return current_loss1_mean, current_loss2_mean

    def batch_train(self, model1, model2, batch_imgs, batch_labels, idx):
        batch_imgs = batch_imgs.to(device=self.device1)
        batch_labels = batch_labels.to(device=self.device1)
        batch_pred1 = model1(batch_imgs)
        batch_imgs = batch_imgs.to(device=self.device2)
        batch_pred2 = model2(batch_imgs)

        corrected_label = self.correct_label(batch_pred1, batch_pred2, batch_labels)
        batch_labels.detach().cpu()

        if (idx + 1) % self.accumulation_step == 0:
            self.optimizer1.zero_grad()

        loss1 = self.loss_fn(batch_pred1, corrected_label) / self.accumulation_step
        loss1.backward()

        if (idx + 1) % self.accumulation_step == 0:
            self.optimizer1.step()
            self.optimizer2.zero_grad()

        corrected_label = corrected_label.to(self.device2)
        loss2 = self.loss_fn(batch_pred2, corrected_label) / self.accumulation_step
        loss2.backward()
        if (idx + 1) % self.accumulation_step == 0:
            self.optimizer2.step()
        # with amp.scale_loss(loss, self.optimizer, loss_id=0) as scaled_loss:
        #     scaled_loss.backward()
        return loss1, loss2

    def valid_epoch(self, model1, model2, loader):
        tqdm_loader = tqdm(loader)
        # current_loss_mean = 0.
        for idx, batch in enumerate(tqdm_loader):
            image, target = batch['image'], batch['label'].cuda()
            with torch.no_grad():
                pred1, pred2 = self.batch_valid(model1, model2, image)
            # loss = self.loss_fn(pred, target)
            # current_loss_mean = (current_loss_mean * idx + loss.item()) / (idx + 1)

            pred1 = pred1.data.cpu().numpy()
            pred2 = pred2.data.cpu().numpy()
            target = target.cpu().numpy()
            pred1 = np.argmax(pred1, axis=1)
            pred2 = np.argmax(pred2, axis=1)
            self.evaluator1.add_batch(target, pred1)
            self.evaluator2.add_batch(target, pred2)

            # tqdm_loader.set_description(f'loss: {current_loss_mean:.4f}')

        # return current_loss_mean

    def batch_valid(self, model1, model2, batch_imgs):
        batch_imgs = batch_imgs.to(device=self.device1)
        batch_pred1 = model1(batch_imgs)
        batch_imgs = batch_imgs.to(device=self.device2)
        batch_pred2 = model2(batch_imgs)
        return batch_pred1, batch_pred2

    def process_summary(self):
        Acc1 = self.evaluator1.Pixel_Accuracy()
        MIoU1 = self.evaluator1.Mean_Intersection_over_Union()
        FWIoU1 = self.evaluator1.Frequency_Weighted_Intersection_over_Union()
        Acc2 = self.evaluator2.Pixel_Accuracy()
        MIoU2 = self.evaluator2.Mean_Intersection_over_Union()
        FWIoU2 = self.evaluator2.Frequency_Weighted_Intersection_over_Union()

        epoch_summary = pd.DataFrame(
            data=[[self.epoch, MIoU1, Acc1, FWIoU1, MIoU2, Acc2, FWIoU2]],
            columns=['epoch', 'MIoU1', 'Acc1', 'FWIoU1', 'MIoU2', 'Acc2', 'FWIoU2']
        )

        if self.distrib_config['LOCAL_RANK'] == 0:
            self.logger.info(f'Epoch {self.epoch}: \t MIoU1: {MIoU1:.6f},'
                             f' Acc1: {Acc1:.6f}, FWIoU1: {FWIoU1:.6f},'
                             f'MIoU2: {MIoU2:.6f},'
                             f' Acc2: {Acc2:.6f}, FWIoU2: {FWIoU2:.6f},')
            self.tb_logger.add_scalar('Valid/MIoU1', MIoU1, self.epoch)
            self.tb_logger.add_scalar('Valid/Acc1', Acc1, self.epoch)
            self.tb_logger.add_scalar('Valid/FWIoU1', FWIoU1, self.epoch)
            self.tb_logger.add_scalar('Valid/MIoU2', MIoU2, self.epoch)
            self.tb_logger.add_scalar('Valid/Acc2', Acc2, self.epoch)
            self.tb_logger.add_scalar('Valid/FWIoU2', FWIoU2, self.epoch)

            if not self.summary_file.is_file():
                epoch_summary.to_csv(self.summary_file, index=False)
            else:
                summary = pd.read_csv(self.summary_file)
                summary = summary.append(epoch_summary)
                summary.to_csv(self.summary_file, index=False)

        return (MIoU1 + MIoU2) / 2

    @staticmethod
    def get_state_dict(model):
        if type(model) == torch.nn.DataParallel:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        return state_dict

    def post_processing(self, score, model1, model2):
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = self.epoch
            if self.distrib_config['LOCAL_RANK'] == 0:
                torch.save(self.get_state_dict(model1), self.best_checkpoint_path1)
                torch.save(self.get_state_dict(model2), self.best_checkpoint_path2)
                self.logger.info('best model: {} epoch - {:.5}'.format(self.epoch, score))

        # if self.distrib_config['LOCAL_RANK'] == 0:
        #     if self.score_heap[0][0] < score:
        #         checkpoints_history_path = Path(
        #             self.checkpoints_history_folder,
        #             '{}_epoch{}.pth'.format(self.calculation_name, self.epoch)
        #         )
        #         torch.save(self.get_state_dict(model), checkpoints_history_path)
        #         heapq.heappush(self.score_heap, (score, checkpoints_history_path))
        #         if len(self.score_heap) > self.checkpoints_topk:
        #             _, removing_checkpoint_path = heapq.heappop(self.score_heap)
        #             removing_checkpoint_path.exists() and removing_checkpoint_path.unlink()
        #             self.logger.info('Removed checkpoint is {}'.format(removing_checkpoint_path))

        if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            self.scheduler.step(score)
        else:
            self.scheduler.step()

    def run_train(self, model1, model2, train_dataloader, valid_dataloader):
        # pdb.set_trace()
        model1.to(self.device1)
        model2.to(self.device2)
        # model, self.optimizer = amp.initialize(model, self.optimizer, opt_level='O1')
        for self.epoch in range(self.n_epoches):
            if self.distrib_config['LOCAL_RANK'] == 0:
                self.logger.info(f'Epoch {self.epoch}: \t start training....')
                self.evaluator1.reset()
                self.evaluator2.reset()
            model1.train()
            model2.train()
            train_loss1_mean, train_loss2_mean = self.train_epoch(model1, model2, train_dataloader)
            if self.distrib_config['LOCAL_RANK'] == 0:
                self.logger.info(f'Epoch {self.epoch}: \t Calculated train loss: {train_loss1_mean:.5f},'
                                 f' {train_loss2_mean:.5f}')
                self.tb_logger.add_scalar('Train/Loss1', train_loss1_mean)
                self.tb_logger.add_scalar('Train/Loss2', train_loss2_mean)

            if self.distrib_config['LOCAL_RANK'] == 0:
                self.logger.info(f'Epoch {self.epoch}: \t start validation....')
            model1.eval()
            model2.eval()
            self.valid_epoch(model1, model2, valid_dataloader)
            selected_score = self.process_summary()

            self.post_processing(selected_score, model1, model2)

            if self.epoch - self.best_epoch > self.early_stopping:
                if self.distrib_config['LOCAL_RANK'] == 0:
                    self.logger.info('EARLY STOPPING')
                break

        if self.distrib_config['LOCAL_RANK'] == 0:
            self.tb_logger.close()

        # self.inference(model, valid_dataloader)
        empty_cuda_cache()
        return self.best_epoch, self.best_score
