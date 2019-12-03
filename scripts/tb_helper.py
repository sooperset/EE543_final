from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import cv2
import torch
import numpy as np
import random
import shutil
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def init_tb_logger(directory, log_file_name):
    log_path = Path(directory, log_file_name)
    if log_path.exists():
        shutil.rmtree(log_path, ignore_errors=True)
    tb_logger = SummaryWriter(log_dir=log_path)
    return tb_logger


def normalize_tensor2numpy(img):
    img = img.permute(1, 2, 0).numpy()
    img = img + abs(img.min())
    img = img / img.max()
    return (img * 255).astype(np.uint8)


def normalize_tensor2tensor(img):
    img = img + img.abs().min()
    img = img / img.max()
    return img * 255


def show_make_image(img, mask, palet):
    # palet = [(0, 0, 0), (249, 192, 12), (0, 185, 241), (114, 0, 218), (249, 50, 12)]

    painted = img
    for ch in range(1,len(palet)):
        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            painted = cv2.polylines(painted, contours[i], True, palet[ch], 2)
    if type(painted) is np.ndarray:
        return painted
    else:
        return cv2.UMat.get(painted)


def make_comparison(img, mask, pred, threshold=0.5):
    palet = [(249, 50, 12)]

    mask = mask.permute(1,2,0).numpy().astype(np.uint8)
    thresholded_pred = (pred > threshold).permute(1, 2, 0).numpy().astype(np.uint8)
    img = normalize_tensor2numpy(img.cpu())

    predicted_img = show_make_image(img, thresholded_pred, palet)
    gt_img = show_make_image(img, mask, palet)
    # heatmap = normalize_tensor2numpy(torch.sum(pred[1:], dim=0).unsqueeze(0).repeat(3, 1, 1))
    blank = np.zeros((10, gt_img.shape[1], gt_img.shape[2]))

    # concat_img = np.concatenate([predicted_img, blank, gt_img, blank, heatmap], axis=0)
    concat_img = np.concatenate([predicted_img, blank, gt_img], axis=0)
    concat_img = torch.tensor(concat_img).permute(2, 0, 1)
    return concat_img


def concat_multiple_comparison(list_comparison):
    return torch.cat(list_comparison, dim=2)


def simple_result(mask, pred):
    mask = np.repeat(mask.permute(1,2,0).numpy().astype(np.uint8), 3, axis=2)
    pred = np.repeat(pred.permute(1,2,0).numpy().astype(np.uint8), 3, axis=2)

    blank = np.ones((10, mask.shape[1], mask.shape[2]))
    concat_img = np.concatenate([mask, blank, pred], axis=0)
    concat_img = torch.tensor(concat_img).permute(2, 0, 1)

    return concat_img

