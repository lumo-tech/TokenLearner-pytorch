from lumo import DatasetBuilder
from lumo.contrib.data.splits import semi_split
from torch.utils.data import RandomSampler

from augmentations.strategies import (standard_multi_crop, standard, randaugment, simclr, read, basic, none,
                                      simclr_randmask)
from .const import mean_std_dic, imgsize_dic, lazy_load_ds
from .datas import pick_datas


def get_train_loader(dataset_name, batch_size=64, batch_count=1024, n_percls=40, mu=7, method='fixmatch',
                     k_size=5, split='train'):
    xs, ys = pick_datas(dataset_name, split=split)

    indice_x, indice_un, _ = semi_split(ys, n_percls=n_percls, val_size=0, include_sup=True, repeat_sup=False)
    print(indice_x[:10], indice_un[:10])
    print(len(set(indice_x)), len(set(indice_un)))
    mean, std = mean_std_dic.get(dataset_name, mean_std_dic.get('default'))
    img_size = imgsize_dic(dataset_name)
    assert img_size is not None

    lazy_load = dataset_name in lazy_load_ds

    sup_ds = (
        DatasetBuilder()
            .add_ids('id')
            .add_input('xs', xs)
            .add_input('ys', ys)
            .add_output('ys', 'ys')
            .subset(indice_x)
    )
    un_ds = (
        DatasetBuilder()
            .add_ids('id')
            .add_input('xs', xs)
            .add_input('ys', ys)
            .add_output('ys', 'ys')
            .subset(indice_un)
    )

    if lazy_load:
        sup_ds.add_input_transform('xs', read)
        un_ds.add_input_transform('xs', read)

    sup_ds.add_output('xs', 'xs', standard(mean, std, size=img_size))

    if method == 'mixmatch':
        for i in range(k_size):
            un_ds.add_output('xs', f'xs{i}', standard_multi_crop(mean, std, size=img_size,
                                                                 index=i))
    elif method in {'fixmatch', 'flexmatch'}:
        (
            un_ds
                .add_ids('ids')
                .add_output('xs', 'xs', standard(mean, std, size=img_size))
                .add_output('xs', 'sxs', randaugment(mean, std, size=img_size))
        )
    elif method == 'comatch':
        (
            sup_ds
                .add_output('xs', 'sxs0', randaugment(mean, std, size=img_size))
                .add_output('xs', 'sxs1', simclr(mean, std, size=img_size))
        )
        (
            un_ds
                .add_output('xs', 'xs', standard(mean, std, size=img_size))
                .add_output('xs', 'sxs0', simclr(mean, std, size=img_size))
                .add_output('xs', 'sxs1', randaugment(mean, std, size=img_size))
        )
    else:  # for default experiments
        un_ds.add_output('xs', 'xs', standard(mean, std, size=img_size))
        (
            sup_ds
                .add_output('xs', 'xs1', standard(mean, std, size=img_size))
                .add_output('xs', 'sxs0', simclr(mean, std, size=img_size))
                .add_output('xs', 'sxs1', simclr(mean, std, size=img_size))
                .add_output('xs', 'sxs2', randaugment(mean, std, size=img_size))
                .add_output('xs', 'sxs3', randaugment(mean, std, size=img_size))
        )
        (
            un_ds
                .add_output('xs', 'xs1', standard(mean, std, size=img_size))
                .add_output('xs', 'sxs0', simclr(mean, std, size=img_size))
                .add_output('xs', 'sxs1', simclr(mean, std, size=img_size))
                .add_output('xs', 'sxs2', randaugment(mean, std, size=img_size))
                .add_output('xs', 'sxs3', randaugment(mean, std, size=img_size))
        )

    sup_dl = sup_ds.DataLoader(batch_size=batch_size,
                               sampler=RandomSampler(data_source=sup_ds,
                                                     replacement=True,
                                                     num_samples=batch_count * batch_size),
                               num_workers=8,
                               pin_memory=True)
    un_dl = un_ds.DataLoader(batch_size=batch_size * mu,
                             sampler=RandomSampler(data_source=un_ds,
                                                   replacement=True,
                                                   num_samples=batch_count * batch_size * mu),
                             num_workers=8,
                             pin_memory=True)
    return sup_dl, un_dl


def get_test_loader(dataset_name):
    xs, ys = pick_datas(dataset_name, split='test')

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

    ds.add_output('xs', 'xs', basic(mean, std, size=img_size))
    ds.add_output('xs', 'xsn', none(mean, std, size=img_size))

    dl = ds.DataLoader(batch_size=128,
                       num_workers=8,
                       pin_memory=True,
                       drop_last=False)

    return dl
