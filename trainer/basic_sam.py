"""

"""
from functools import partial

import torch
from lumo.contrib import EMA
from lumo.kit.trainer import TrainerResult
from torch import nn
from torch.nn import functional as F

from models.module_utils import (pick_model_name,
                                 ViTOutputs)
from .suptrainer import *
from torch.optim.sgd import SGD
from models.sam import SAM


class BasicCEParams(SupParams):

    def __init__(self):
        super().__init__()
        self.aug_type = self.choice('basic', 'simclr', 'randaug')
        self.model = 'vit'


ParamsType = BasicCEParams


class BasicCEModule(nn.Module):

    def __init__(self, model_name,
                 n_classes=10):
        super().__init__()
        self.backbone = pick_model_name(model_name)
        input_dim = self.backbone.feature_dim
        self.classifier = nn.Linear(input_dim, n_classes)

    def forward(self, xs):
        feature_map = self.backbone(xs)
        logits = self.classifier(feature_map)

        output = ViTOutputs()
        output.feature_map = feature_map
        output.logits = logits
        return output


class BasicCETrainer(SupTrainer):

    def to_logits(self, xs):
        if self.params.ema:
            return self.ema_model.forward(xs).logits
        return self.model.forward(xs).logits

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = BasicCEModule(params.model,
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
            self.optim = SAM(param_list, SGD, lr=params.optim.lr, momentum=0.9)
        else:
            self.optim = SAM(self.model.parameters(), SGD, lr=params.optim.lr, momentum=0.9)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=params.epoch)

        self.to_device()

        if params.ema:
            self.ema_model = EMA(self.model, alpha=0.999)

    def on_train_epoch_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        super().on_train_epoch_end(trainer, func, params, result, *args, **kwargs)

    def train_step(self, idx, batch, params: ParamsType, *args, **kwargs) -> Meter:
        meter = Meter()

        ys = batch['ys']
        if params.aug_type == 'basic':
            xs = batch['xs']
        elif params.aug_type == 'simclr':
            xs = batch['sxs0']
        elif params.aug_type == 'randaug':
            xs = batch['sxs1']
        else:
            raise NotImplementedError()

        output = self.model.forward(xs)
        Lx = F.cross_entropy(output.logits, ys)
        Lx.backward()
        self.optim.first_step(zero_grad=True)

        output = self.model.forward(xs)
        Lx = F.cross_entropy(output.logits, ys)
        Lx.backward()
        self.optim.second_step(zero_grad=True)

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


main = partial(main, BasicCETrainer, ParamsType)
