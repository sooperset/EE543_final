import warnings
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import f1_score, precision_score
import torch
import cv2
import pdb
warnings.filterwarnings("ignore")

### Segmentation ###
def search_deep_thresholds(eval_list, thrs_list, n_search_workers):
    best_score = 0.
    best_thrs = []
    # pdb.set_trace()
    progress_bar = tqdm(thrs_list)

    for thrs in progress_bar:
        thr = thrs[0]
        score_list = Parallel(n_jobs=n_search_workers)(delayed(apply_deep_thresholds)(
            probas, labels, thr) for probas, labels in eval_list)
        final_score = np.mean(score_list)
        if final_score > best_score:
            best_score = final_score
            best_thrs = thrs
        progress_bar.set_description('Best score: {:.4}'.format(best_score))
    return best_score, best_thrs


def apply_deep_thresholds(predicted, ground_truth, threshold=0.3):
    mask = predicted > threshold
    return dice_fn(mask, ground_truth)


def dice_fn(mask, truth):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    with torch.no_grad():
        mask = mask.reshape(batch_size, -1).float()
        truth = truth.reshape(batch_size, -1).float()
        assert(mask.shape == truth.shape)\

        t_sum = truth.sum(-1)
        m_sum = mask.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (m_sum == 0).float()
        dice_pos = 2 * (mask*truth).sum(-1)/((mask+truth).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice # , dice_neg, dice_pos, num_neg, num_pos


''' ### numpy
def dice_fn(predicted, ground_truth):
    eps = 1e-4
    batch_size = predicted.shape[0]

    predicted = predicted.reshape(batch_size, -1).astype(np.bool)
    ground_truth = ground_truth.reshape(batch_size, -1).astype(np.bool)

    intersection = np.logical_and(predicted, ground_truth).sum(axis=1)
    union = predicted.sum(axis=1) + ground_truth.sum(axis=1) + eps
    loss = (2. * intersection + eps) / union
    return loss.mean()
'''

''' ### Too slow..
def apply_deep_thresholds_component_area(predicted, ground_truth, threshold=0.5, area_threshold=0):
    mask = predicted.copy()
    thresholded_mask = mask > threshold
    predictions = np.zeros_like(mask, dtype=np.float32)
    for batch_idx in range(thresholded_mask.shape[0]):
        for ch in range(thresholded_mask.shape[1]):
            num_component, component = cv2.connectedComponents(thresholded_mask[batch_idx][ch].astype(np.uint8))
            for c in range(1, num_component):
                p = (component == c)
                if p.sum() > area_threshold:
                    predictions[batch_idx][ch][p] = 1
    return dice_fn(predictions, ground_truth)
'''