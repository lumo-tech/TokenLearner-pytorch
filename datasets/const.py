mean_std_dic = {
    'cifar10': [(0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)],
    'cifar100': [(0.5071, 0.4867, 0.4408),
                 (0.2675, 0.2565, 0.2761)],
    'svhn': [(0.44921358705286946, 0.4496640988868895, 0.45029627318846444),
             (0.2003216966442779, 0.1991626631851053, 0.19936594996908613)],
    'stl10': [(0.44319644335512015, 0.4463139215686274, 0.44558495098039186),
              (0.26640707592603363, 0.2644222394907146, 0.2637064714107059)],
    'default': [(0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)],

}

# from https://github.com/tjmoon0104/pytorch-tiny-imagenet/blob/master/ResNet18_224.ipynb
for k in ['tinyimagenet', 'tinyimagenet-64']:
    mean_std_dic[k] = [(0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)]

# from https://github.com/pytorch/examples/blob/master/imagenet/main.py
for k in ['imagenet', 'imagenet-64', 'imagenet-96', 'imagenet-112']:
    mean = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]

imgsize_dic_ = {
    'cifar10': 32,
    'cifar10-64': 64,
    'cifar10-96': 96,
    'cifar100': 32,
    'cifar100-64': 32,
    'cifar100-96': 32,
    'stl10': 96,
    'svhn': 32,
    'tinyimagenet': 72,
    'tinyimagenet-64': 64,
    'imagenet': 224,
    'imagenet-64': 64,
    'imagenet-96': 96,
    'imagenet-112': 112,
}


def imgsize_dic(dataset_name):
    res = dataset_name.split('-')
    assert res[0] in imgsize_dic_
    if len(res) == 2:
        return int(res[1])
    return imgsize_dic_[res[0]]


lazy_load_ds = {
    'tinyimagenet',
    'tinyimagenet-64',
    'imagenet',
    'imagenet-64',
    'imagenet-96',
    'imagenet-112',
    'clothing1m'
}

n_classes = {
    'cifar10': 10,
    'cifar10-64': 10,
    'cifar10-96': 10,
    'cifar100': 100,
    'cifar100-64': 100,
    'cifar100-96': 100,
    'svhn': 10,
    'stl10': 10,
    'tinyimagenet': 200,
    'tinyimagenet-64': 200,
    'imagenet': 1000,
    'imagenet-64': 1000,
    'imagenet-96': 1000,
    'imagenet-112': 1000,
}
