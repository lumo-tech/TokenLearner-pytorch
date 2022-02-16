"""
refer to
 - https://github.com/kuangliu/pytorch-cifar
"""
from typing import ClassVar

from lumo import Trainer, Params, Meter, callbacks, DataModule
from lumo.kit.beans.trainstage import TrainerStage

from datasets.supdataset import get_train_loader, get_test_loader
from models.module_utils import ModelParams
from datasets.dataset_utils import DataParams


class SupParams(Params, ModelParams, DataParams):

    def __init__(self):
        super().__init__()
        self.seed = 1
        self.method = None

        self.epoch = 200
        self.batch_size = 128
        self.optim = self.OPTIM.create_optim('SGD', lr=0.01, weight_decay=5e-4, momentum=0.9)

        self.ema = True

        self.warmup_epochs = 10
        self.pretrain_path = None


ParamsType = SupParams


class SupTrainer(Trainer, callbacks.TrainCallback, callbacks.InitialCallback):

    def on_train_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        super().on_train_begin(trainer, func, params, *args, **kwargs)
        self.rnd.mark(params.seed)
        self.logger.info(f'set seed {params.seed}')

    def on_prepare_dataloader_end(self, trainer: Trainer, func, params: ParamsType, meter: Meter, *args, **kwargs):
        super().on_prepare_dataloader_end(trainer, func, params, meter, *args, **kwargs)
        res = self.getfuncargs(func, *args, **kwargs)
        stage = res['stage'].name
        if stage == 'train':
            if params.warmup_epochs > 0:
                self.lr_sche = params.SCHE.List([
                    params.SCHE.Linear(
                        start=1e-7, end=params.optim.lr,
                        left=0,
                        right=len(self.train_dataloader) * params.warmup_epochs
                    ),
                    params.SCHE.Cos(
                        start=params.optim.lr, end=1e-7,
                        left=len(self.train_dataloader) * params.warmup_epochs,
                        right=len(self.train_dataloader) * params.epoch
                    ),
                ])
            else:
                self.lr_sche = params.SCHE.Cos(
                    start=params.optim.lr, end=1e-7,
                    left=0,
                    right=len(self.train_dataloader) * params.epoch
                )
            self.logger.info('create learning scheduler')
            self.logger.info(self.lr_sche)

    def icallbacks(self, params: ParamsType):
        super().icallbacks(params)
        callbacks.EvalCallback(eval_per_epoch=-1, test_per_epoch=1).hook(self)
        callbacks.LoggerCallback(step_frequence=1, breakline_in=150).hook(self)
        callbacks.AutoLoadModel().hook(self)
        callbacks.ScalarRecorder().hook(self)
        if isinstance(self, callbacks.BaseCallback):
            self.hook(self)

    def to_logits(self, xs):
        raise NotImplementedError()

    def test_step(self, idx, batch, params: ParamsType, *args, **kwargs) -> Meter:
        meter = Meter()
        xs0 = batch['xs0']
        xs1 = batch['xs1']
        ys = batch['ys']
        logits0 = self.to_logits(xs0)
        logits1 = self.to_logits(xs1)

        meter.sum.Acc0 = (logits0.argmax(dim=-1) == ys).sum()
        meter.sum.Acc1 = (logits1.argmax(dim=-1) == ys).sum()
        meter.sum.C = xs0.shape[0]
        return meter

    @property
    def metric_step(self):
        return self.global_step % 100 == 0


class SupDM(DataModule):
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



def main(trainer_cls: ClassVar[Trainer], params_cls: ClassVar[ParamsType], dm: DataModule = None):
    params = params_cls()
    params.from_args()
    trainer = trainer_cls(params)

    if dm is None:
        dm = SupDM()

    if params.pretrain_path is not None and params.train_linear:
        trainer.load_state_dict(params.pretrain_path)
        trainer.test(dm)
        return

    trainer.rnd.mark(params.seed)
    trainer.train(dm)
    trainer.save_model()
