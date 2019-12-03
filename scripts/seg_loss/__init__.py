from .boundary import BDLoss, DC_and_BD_loss, HDDTBinaryLoss, DC_and_HDBinary_loss, DistBinaryDiceLoss
from .focal import FocalLoss
from .dice import (GDiceLoss, GDiceLossV2, SoftDiceLoss, SSLoss, IoULoss, TverskyLoss, FocalTversky_loss, AsymLoss,
                   DC_and_CE_loss, PenaltyGDiceLoss, DC_and_topk_loss, ExpLog_loss)
from .Lovasz import LovaszSoftmax

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

all = [BDLoss, DC_and_BD_loss, HDDTBinaryLoss, DC_and_HDBinary_loss, DistBinaryDiceLoss,
       FocalLoss, GDiceLoss, GDiceLossV2, SoftDiceLoss, SSLoss, IoULoss, TverskyLoss, FocalTversky_loss, AsymLoss,
       DC_and_CE_loss, PenaltyGDiceLoss, DC_and_topk_loss, ExpLog_loss,
       LovaszSoftmax, BCEWithLogitsLoss, CrossEntropyLoss]

losses = {}
for loss_class in all:
    losses.update({loss_class.__name__: loss_class})

def get_loss(name, **args):
    # print(losses)
    if 'pos_weight' in args:
        args['pos_weight'] = torch.tensor(args['pos_weight'], dtype=torch.float).cuda()
    if 'weight' in args:
        args['weight'] = torch.tensor(args['weight'], dtype=torch.float).cuda()

    loss_class = losses[name]
    loss_func = loss_class(**args)
    return loss_func
