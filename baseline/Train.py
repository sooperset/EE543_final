import argparse
import os
import sys
from pathlib import Path
from scripts.utils import load_yaml, seed_everything, init_logger, WrappedModel, DistributedWeightedRandomSampler
from scripts.tb_helper import init_tb_logger
from scripts.metric import apply_deep_thresholds, search_deep_thresholds
from scripts.seg_loss import get_loss
from baseline.Learning import Learning
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import torch.nn as nn
from torchvision.datasets import VOCSegmentation
import albumentations as albu
import functools
import importlib
from ast import literal_eval
from apex.parallel import DistributedDataParallel, convert_syncbn_model

sys.path.append('/workspace/lib/segmentation_models.pytorch')
sys.path.append('/workspace/lib/utils')
import radam

class SingleFromMultipleLoadError(Exception):
    def __str__(self):
        return "SingleFromMultipleLoadError"

class MultipleFromSingleLoadError(Exception):
    def __str__(self):
        return "MultipleFromSingleLoadError"

def argparser():
    parser = argparse.ArgumentParser(description='VOC Segmentation')
    parser.add_argument('--config_path', type=str, help='train config path')
    parser.add_argument('--local_rank', type=int, default=0, help='for distributed')
    return parser.parse_args()


def init_eval_fns(train_config):
    search_pairs = train_config['EVALUATION']['SEARCH_PAIRS']
    n_search_workers = train_config.get('WORKERS', 1)
    # n_search_workers = -2

    thrs_list = [thr for thr in search_pairs]

    main_thr = 0.5
    print('Evaluation thresholds: ', main_thr)
    local_metric_fn = functools.partial(
        apply_deep_thresholds,
        threshold=0.5
    )

    global_metric_fn = functools.partial(
        search_deep_thresholds,
        thrs_list=thrs_list,
        n_search_workers=n_search_workers
    )
    return local_metric_fn, global_metric_fn


def train_fold(
    train_config, distrib_config, pipeline_name, log_dir, fold_id,
    train_dataloader, valid_dataloader,
    local_metric_fn, global_metric_fn):

    if distrib_config['LOCAL_RANK'] == 0:
        fold_logger = init_logger(log_dir, 'train_fold_{}.log'.format(fold_id))
        fold_tb_logger = init_tb_logger(log_dir, 'train_fold_{}'.format(fold_id))

    best_checkpoint_folder = Path(log_dir, train_config['CHECKPOINTS']['BEST_FOLDER'])
    best_checkpoint_folder.mkdir(exist_ok=True, parents=True)

    checkpoints_history_folder = Path(
        log_dir,
        train_config['CHECKPOINTS']['FULL_FOLDER'],
        'fold{}'.format(fold_id)
    )
    checkpoints_history_folder.mkdir(exist_ok=True, parents=True)
    checkpoints_topk = train_config['CHECKPOINTS']['TOPK']

    calculation_name = '{}_fold{}'.format(pipeline_name, fold_id)

    device = train_config['DEVICE']

    if train_config['MODEL']['ARGS']['activation'] is not None:
        train_config['MODEL']['ARGS']['activation'] = None
    module = importlib.import_module(train_config['MODEL']['PY'])
    model_function = getattr(module, train_config['MODEL']['CLASS'])
    model_args = train_config['MODEL']['ARGS']
    if 'Unet' in train_config['MODEL']['CLASS']:
        if type(model_args['decoder_channels']) == str:
            model_args['decoder_channels'] = literal_eval(model_args['decoder_channels'])
    model_base = model_function(**train_config['MODEL']['ARGS'])
    model = model_function(**train_config['MODEL']['ARGS'])

    if len(train_config['DEVICE_LIST']) > 1:
        model.cuda()
        model = convert_syncbn_model(model)
        model = DistributedDataParallel(model, delay_allreduce=True)

    pretrained_model_path = best_checkpoint_folder / f'{calculation_name}.pth'
    if pretrained_model_path.is_file():
        try:  # single gpu <- single gpu , multiple gpu <- multiple gpu
            state_dict = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)
        except SingleFromMultipleLoadError:  # single gpu <- multiple gpu
            model = WrappedModel(model)  # to load a model which is saved from parallel learning
            state_dict = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)
        except MultipleFromSingleLoadError:  # multiple gpu <- single gpu
            model = model_base
            state_dict = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)
            model.cuda()
            model = convert_syncbn_model(model)
            model = DistributedDataParallel(model, delay_allreduce=True)

        if distrib_config['LOCAL_RANK'] == 0:
            fold_logger.info('load model from {}'.format(pretrained_model_path))

    if 'ARGS' in train_config['CRITERION']:
        loss_fn = get_loss(train_config['CRITERION']['NAME'], **train_config['CRITERION']['ARGS'])
    else:
        loss_fn = get_loss(train_config['CRITERION']['NAME'])

    if train_config['OPTIMIZER']['CLASS'] == 'RAdam':
        optimizer_class = getattr(radam, train_config['OPTIMIZER']['CLASS'])
    else:
        optimizer_class = getattr(torch.optim, train_config['OPTIMIZER']['CLASS'])
    optimizer = optimizer_class(model.parameters(), **train_config['OPTIMIZER']['ARGS'])

    scheduler_class = getattr(torch.optim.lr_scheduler, train_config['SCHEDULER']['CLASS'])
    scheduler = scheduler_class(optimizer, **train_config['SCHEDULER']['ARGS'])

    n_epoches = train_config['EPOCHS']
    accumulation_step = train_config['ACCUMULATION_STEP']
    early_stopping = train_config['EARLY_STOPPING']

    if distrib_config['LOCAL_RANK'] != 0:
        fold_logger = None
        fold_tb_logger = None

    best_epoch, best_score = Learning(
        distrib_config,
        optimizer,
        loss_fn,
        device,
        n_epoches,
        scheduler,
        accumulation_step,
        early_stopping,
        fold_logger,
        fold_tb_logger,
        best_checkpoint_folder,
        checkpoints_history_folder,
        checkpoints_topk,
        calculation_name
    ).run_train(model, train_dataloader, valid_dataloader, local_metric_fn, global_metric_fn)

    fold_logger.info(f'Best Epoch : {best_epoch}, Best Score : {best_score}')


if __name__ == '__main__':
    args = argparser()
    train_config = load_yaml(args.config_path)
    distrib_config = {}
    distrib_config['LOCAL_RANK'] = args.local_rank

    root_dir = Path(train_config['DIRECTORY']['ROOT_DIRECTORY'])
    data_dir = Path(train_config['DIRECTORY']['DATA_DIRECTORY'])
    mask_dir = Path(train_config['DIRECTORY']['MASK_DIRECTORY'])
    log_dir = root_dir / train_config['DIRECTORY']['LOGGER_DIRECTORY']
    log_dir.mkdir(exist_ok=True, parents=True)

    if distrib_config['LOCAL_RANK'] == 0:
        main_logger = init_logger(log_dir, 'train_main.log')

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    SEED = train_config['SEED']
    seed_everything(SEED)
    if distrib_config['LOCAL_RANK'] == 0:
        main_logger.info(train_config)

    if "DEVICE_LIST" in train_config:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, train_config["DEVICE_LIST"]))

    if len(train_config['DEVICE_LIST']) > 1:
        distrib_config['DISTRIBUTED'] = True
        torch.cuda.set_device(distrib_config['LOCAL_RANK'])
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        distrib_config['WORLD_SIZE'] = torch.distributed.get_world_size()
        train_config['OPTIMIZER']['ARGS']['lr'] = train_config['OPTIMIZER']['ARGS']['lr'] * float(
            train_config['BATCH_SIZE'] * distrib_config['WORLD_SIZE']) / 256
    else:
        distrib_config['DISTRIBUTED'] = False
        distrib_config['WORLD_SIZE'] = False

    pipeline_name = train_config['PIPELINE_NAME']

    train_transform = albu.load(train_config['TRANSFORMS']['TRAIN'])
    valid_transform = albu.load(train_config['TRANSFORMS']['VALID'])

    folds_distr_path = Path(train_config['FOLD']['FILE'])

    num_workers = train_config['WORKERS']
    batch_size = train_config['BATCH_SIZE']
    n_folds = train_config['FOLD']['NUMBER']

    usefolds = map(str, train_config['FOLD']['USEFOLDS'])
    local_metric_fn, global_metric_fn = init_eval_fns(train_config)

    for fold_id in usefolds:
        if distrib_config['LOCAL_RANK'] == 0:
            main_logger.info('Start training of {} fold....'.format(fold_id))

        train_dataset = CellDataset(data_dir, mask_dir, folds_distr_path, n_folds, fold_id, train_transform, 'train')
        valid_dataset = CellDataset(data_dir, mask_dir, folds_distr_path, n_folds, fold_id, valid_transform, 'valid')

        if len(train_config['DEVICE_LIST']) > 1:
            if train_config['USE_SAMPLER']:
                counts = np.unique(train_dataset.df.defects.values, return_counts=True)[1]
                weights = np.sum(counts) / counts
                weights = torch.DoubleTensor(weights)
                train_sampler = DistributedWeightedRandomSampler(train_dataset, weights)
                valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
            else:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)

            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers,
                                      shuffle=False, sampler=train_sampler, pin_memory=True)
            valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, num_workers=num_workers,
                                      shuffle=False, sampler=valid_sampler, pin_memory=True)
        else:
            if train_config['USE_SAMPLER']:
                counts = np.unique(train_dataset.df.defects.values, return_counts=True)[1]
                weights = np.sum(counts) / counts
                samples_weights = torch.DoubleTensor([weights[t-1] for t in train_dataset.df.defects.values])
                train_sampler = WeightedRandomSampler(weights, len(train_dataset))
            else:
                train_sampler = None
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers,
                                      shuffle=(train_sampler is None), sampler=train_sampler, pin_memory=True)
            valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, num_workers=num_workers,
                                      shuffle=False, pin_memory=True)

        train_fold(
            train_config, distrib_config, pipeline_name, log_dir,
            fold_id, train_loader, valid_loader,
            local_metric_fn, global_metric_fn
        )
