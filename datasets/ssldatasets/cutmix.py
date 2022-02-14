import numpy as np
from torch.utils.data import RandomSampler

from augmentations.strategies import standard_multi_crop, standard, randaugment, simclr, read

from ..datas import regist_data
from ..const import mean_std_dic, imgsize_dic, lazy_load_ds
from lumo import DatasetBuilder


def get_train_loader(dataset_name, batch_size=64, batch_count=1024, method='basic'):
    data_fn = regist_data.get(dataset_name, None)
    assert data_fn is not None
    xs, ys = data_fn(train=True)

    mean, std = mean_std_dic.get(dataset_name, mean_std_dic.get('default'))
    img_size = imgsize_dic(dataset_name)
    assert img_size is not None

    lazy_load = dataset_name in lazy_load_ds

    ds = (
        DatasetBuilder()
            .add_ids('id')
            .add_input('xs', xs)
            .add_input('ys', ys)
            .add_output('xs', 'xs', standard(mean, std, size=img_size))
            .add_output('xs', 'sxs0', simclr(mean, std, size=img_size))
            .add_output('xs', 'sxs1', simclr(mean, std, size=img_size))
            .add_output('ys', 'ys')
    )

    if lazy_load:
        ds.add_input_transform('xs', read)

    dl = ds.DataLoader(batch_size=batch_size,
                       sampler=RandomSampler(data_source=ds,
                                             replacement=True,
                                             num_samples=batch_count * batch_size),
                       num_workers=8,
                       pin_memory=True)
    return dl


def get_test_loader(dataset_name):
    data_fn = regist_data.get(dataset_name, None)
    assert data_fn is not None
    xs, ys = data_fn(train=False)

    mean, std = mean_std_dic.get(dataset_name, mean_std_dic.get('default'))
    img_size = imgsize_dic(dataset_name)
    assert img_size is not None

    lazy_load = dataset_name in lazy_load_ds

    ds = (
        DatasetBuilder()
            .add_ids('id')
            .add_input('xs', xs)
            .add_input('ys', ys)
            .add_output('ys', 'ys')
    )

    if lazy_load:
        ds.add_input_transform('xs', read)

    ds.add_output('xs', 'xs', standard(mean, std, size=img_size))

    dl = ds.DataLoader(batch_size=128,
                       num_workers=8,
                       pin_memory=True,
                       drop_last=False)

    return dl
