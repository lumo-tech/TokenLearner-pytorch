"""

"""
from lumo.calculate.tensor import onehot
from lumo.contrib.data.augments.mix import mixup
from lumo.contrib.nn.loss import cross_entropy_with_targets
from .basic import *


class MixupParams(SupParams):

    def __init__(self):
        super().__init__()
        self.aug_type = self.choice('basic', 'simclr', 'randaug')
        self.apply_mix = True
        self.split_params = True
        self.beta = 1.0
        self.mixup_type = self.choice('multihot', 'dual')


ParamsType = MixupParams


class MixupTrainer(BasicCETrainer):

    def train_step(self, idx, batch, params: ParamsType, *args, **kwargs) -> Meter:
        meter = Meter()

        if params.aug_type == 'basic':
            xs = batch['xs']
        elif params.aug_type == 'simclr':
            xs = batch['sxs0']
        elif params.aug_type == 'randaug':
            xs = batch['sxs1']
        else:
            raise NotImplementedError()

        ys = batch['ys']

        if params.apply_mix:
            reid = torch.randperm(len(xs))
            mixed_xs, lam = mixup(xs, xs[reid])
            targets = onehot(ys, params.n_classes)
            mixed_targets = targets * lam + targets[reid] * (1 - lam)
            output = self.model.forward(mixed_xs)
            if params.mixup_type == 'dual':
                Lx = (F.cross_entropy(output.logits, ys) * lam +
                      F.cross_entropy(output.logits, ys[reid]) * (1 - lam))
            elif params.mixup_type == 'multihot':
                Lx = cross_entropy_with_targets(output.logits, mixed_targets)
            else:
                raise NotImplementedError()

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


main = partial(main, MixupTrainer, ParamsType)
