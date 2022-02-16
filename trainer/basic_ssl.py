"""
supervised contrastive loss
"""
from functools import partial
from lumo.contrib.nn.loss import contrastive_loss2

import torch
from lumo.contrib import EMA
from lumo.kit.trainer import TrainerResult
from torch import nn
from torch.nn import functional as F

from models.module_utils import (pick_model_name,
                                 ViTOutputs)
from models.components import MLP2
from .suptrainer import *
from datasets.ssldataset import get_train_loader, get_test_loader


class BasicCEParams(SupParams):

    def __init__(self):
        super().__init__()
        self.aug_type = self.choice('basic', 'simclr', 'randaug')
        self.model = 'vit'
        # self.apply_mixup = False
        # self.split_params = True
        self.apply_sam = True


ParamsType = BasicCEParams


class BasicCEModule(nn.Module):

    def __init__(self, model_name,
                 n_classes=10):
        super().__init__()
        self.backbone = pick_model_name(model_name)
        input_dim = self.backbone.feature_dim
        self.head = MLP2(input_dim, 1024, 128)
        self.classifier = nn.Linear(input_dim, n_classes)

    def forward(self, xs):
        feature_map = self.backbone(xs)
        logits = self.classifier(feature_map)
        output = ViTOutputs()
        output.feature_map = feature_map
        output.feature = self.head(feature_map)
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
        xs0 = batch['sxs0']
        xs1 = batch['sxs1']

        output0 = self.model.forward(xs0)
        output1 = self.model.forward(xs1)
        query = output0.feature
        key = output1.feature
        qk_graph = ys[:, None] == ys[None, :]
        Lcs = contrastive_loss2(query=query, key=key,
                                temperature=params.temperature,
                                norm=True,
                                query_neg=False,
                                qk_graph=qk_graph,
                                eye_one_in_qk=False)
        logits = output0.logits
        Lx = F.cross_entropy(logits, ys)

        self.optim.zero_grad()
        self.accelerator.backward(Lcs + Lx)
        self.optim.step()
        cur_lr = self.lr_sche.apply(self.optim, self.global_step)
        if params.ema:
            self.ema_model.step()

        with torch.no_grad():
            meter.mean.Lcs = Lcs
            meter.mean.Lx = Lx
            meter.mean.Ax = (logits.argmax(dim=-1) == ys).float().mean()
            meter.last.lr = cur_lr

        return meter


class SSLDM(DataModule):
    def idataloader(self, params: ParamsType, stage: TrainerStage, repeat: bool = False):
        super().idataloader(params, stage, repeat)
        if stage.is_train:
            dl = get_train_loader(params.dataset,
                                  params.batch_size,
                                  method=params.method)

        elif stage.is_test:
            dl = get_test_loader(params.dataset)
        else:
            raise NotImplementedError()
        self.regist_dataloader_with_stage(stage, dl)


main = partial(main, BasicCETrainer, ParamsType, SSLDM())
