"""

"""
from functools import partial

import numpy as np
import torch
from lumo.calculate.tensor import onehot
from lumo.contrib import EMA
from lumo.contrib.nn.loss import cross_entropy_with_targets
from lumo.kit.trainer import TrainerResult
from torch import nn
from torch.nn import functional as F

from models.module_utils import (pick_model_name,
                                 ResnetOutput)
from tricks.cutmix import rand_bbox
from .suptrainer import *


class BCEParams(SupParams):

    def __init__(self):
        super().__init__()
        self.aug_type = self.choice('basic', 'simclr', 'randaug')
        self.apply_cutmix = True
        self.split_params = True
        self.beta = 1.0


ParamsType = BCEParams


class BCEModule(nn.Module):

    def __init__(self, model_name,
                 n_classes=10):
        super().__init__()
        self.backbone = pick_model_name(model_name)
        input_dim = self.backbone.feature_dim
        self.classifier = nn.Linear(input_dim, n_classes)

    def forward(self, xs):
        feature_map = self.backbone(xs)
        logits = self.classifier(feature_map)

        output = ResnetOutput()
        output.feature_map = feature_map
        output.logits = logits
        return output


class CutmixTrainer(SupTrainer):

    def to_logits(self, xs):
        if self.params.ema:
            return self.ema_model.forward(xs).logits
        return self.model.forward(xs).logits

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = BCEModule(params.model,
                               n_classes=params.n_classes)

        if params.split_params:
            wd_params, non_wd_params = [], []
            for name, param in self.model.named_parameters():
                if 'bn' in name:
                    non_wd_params.append(param)
                else:
                    wd_params.append(param)
            param_list = [
                {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
            self.optim = params.optim.build(param_list)
        else:
            self.optim = params.optim.build(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=params.epoch)

        self.to_device()

        if params.ema:
            self.ema_model = EMA(self.model, alpha=0.999)

    def on_train_epoch_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        super().on_train_epoch_end(trainer, func, params, result, *args, **kwargs)

    def train_step(self, idx, batch, params: ParamsType, *args, **kwargs) -> Meter:
        meter = Meter()

        ys = batch['ys']
        target = onehot(ys, params.n_classes)

        xs = batch['xs']

        if params.apply_cutmix:
            lam = np.random.beta(params.beta, params.beta)
            reid = torch.randperm(xs.size()[0], device=self.device)
            target_b = target[reid]
            bbx1, bby1, bbx2, bby2 = rand_bbox(xs.size(), lam)
            xs[:, :, bbx1:bbx2, bby1:bby2] = xs[reid, :, bbx1:bbx2, bby1:bby2]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (xs.size()[-1] * xs.size()[-2]))
            output = self.model.forward(xs).logits
            Lx = (cross_entropy_with_targets(output, target) * lam +
                  cross_entropy_with_targets(output, target_b) * (1. - lam))
        else:
            output = self.model.forward(xs)
            Lx = F.cross_entropy(output.logits, ys)

        self.optim.zero_grad()
        self.accelerator.backward(Lx)
        self.optim.step()
        cur_lr = self.lr_sche.apply(self.optim, self.global_step)
        if params.ema:
            self.ema_model.step()

        with torch.no_grad():
            meter.mean.Lx = Lx
            if params.apply_mixup:
                logits = self.model.forward(xs).logits
            else:
                logits = output.logits
            meter.mean.Ax = (logits.argmax(dim=-1) == ys).float().mean()
            meter.last.lr = cur_lr

        return meter


main = partial(main, CutmixTrainer, ParamsType)
