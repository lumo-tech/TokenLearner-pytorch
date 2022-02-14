from lumo import DataBundler, BaseParams
from torch.utils.data.dataloader import DataLoader

from . import const


class DataParams(BaseParams):

    def __init__(self):
        super().__init__()
        self.dataset = self.choice(
            'cifar10',
            'cifar10-64',
            'cifar10-96',
            'cifar100',
            'cifar100-64',
            'cifar100-96',
            'svhn',
            'stl10',
            'tinyimagenet',
            'tinyimagenet-64',
            'imagenet',
            'imagenet-64',
            'imagenet-96',
            'imagenet-112',
        )
        self.n_classes = 10

    def iparams(self):
        super().iparams()
        self.n_classes = const.n_classes[self.dataset]


def summary_dataloader(loader):
    res = []
    if isinstance(loader, DataLoader):
        res.append(loader.dataset)

    elif isinstance(loader, DataBundler):
        for key, (l, func) in loader.dataloaders.items():
            res.append(l.dataset)
    return res
