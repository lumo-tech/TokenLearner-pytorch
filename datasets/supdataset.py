from lumo import DatasetBuilder

from augmentations.strategies import standard, simclr, read, randaugment, basic, none
from .const import mean_std_dic, imgsize_dic, lazy_load_ds
from .datas import pick_datas


def get_train_loader(dataset_name, batch_size=64, method='default', split='train'):
    xs, ys = pick_datas(dataset_name, split=split)

    mean, std = mean_std_dic.get(dataset_name, mean_std_dic.get('default'))
    img_size = imgsize_dic(dataset_name)
    assert img_size is not None
    print(img_size)
    lazy_load = dataset_name in lazy_load_ds

    ds = (
        DatasetBuilder()
            .add_ids('id')
            .add_input('xs', xs)
            .add_input('ys', ys)
            .add_output('xs', 'xs', standard(mean, std, size=img_size))
            .add_output('xs', 'sxs0', simclr(mean, std, size=img_size))
            .add_output('xs', 'sxs1', randaugment(mean, std, size=img_size))
            .add_output('ys', 'ys')
    )

    if lazy_load:
        ds.add_input_transform('xs', read)

    dl = ds.DataLoader(batch_size=batch_size,
                       num_workers=8,
                       shuffle=True,
                       pin_memory=True)
    return dl


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

    ds.add_output('xs', 'xs0', basic(mean, std, size=img_size))
    ds.add_output('xs', 'xs1', none(mean, std, size=img_size))

    dl = ds.DataLoader(batch_size=128,
                       num_workers=8,
                       pin_memory=True,
                       drop_last=False)

    return dl
