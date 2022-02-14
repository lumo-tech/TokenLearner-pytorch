from joblib.memory import Memory
from lumo import llist
from lumo.proc.path import cache_dir
import os
from lumo.utils.filebranch import FileBranch

mem = Memory(location=os.path.join(cache_dir(), 'lumo.joblib'))
# mem.clear()

roots = {
    'tinyimagenet200': '/home/yanghaozhe/datasets/tiny-imagenet/tiny-imagenet-200',
    'imagenet': '/home/yanghaozhe/datasets/imagenet',
}


@mem.cache
def tiny_imagenet200(split='train'):
    """
    download and unzip from http://cs231n.stanford.edu/tiny-imagenet-200.zip

    ```
    cd root
    ls
    >>> test  train  val  wnids.txt  words.txt
    ```
    """
    root = roots['tinyimagenet200']
    if split == 'train':
        xs = list(FileBranch(root).branch('train').find_file_in_depth('.JPEG', 2))
        name_cls_map = {name: i for i, name in enumerate(sorted(FileBranch(root).branch('train').listdir()))}
        ys = [name_cls_map[os.path.basename(x).split('_')[0]] for x in xs]
    else:  # use val dir for testing by default.
        with open(os.path.join(root, 'val', 'val_annotations.txt')) as r:
            lines = r.readlines()
        pairs = [line.split('\t')[:2] for line in lines]
        xs = [os.path.join(root, 'val', 'images', fn) for fn, _ in pairs]
        name_cls_map = {name: i for i, name in enumerate(sorted(set([name for _, name in pairs])))}
        ys = [name_cls_map[name] for _, name in pairs]

    return llist(xs), llist(ys)


@mem.cache
def imagenet(split='train'):
    """
    download from https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description
    ```
    mkdir imagenet
    cd ./imagenet
    kaggle competitions download -c imagenet-object-localization-challenge
    unzip imagenet-object-localization-challenge.zip
    tar -xvf imagenet_object_localization_patched2019.tar.gz
    ls
    >>> ILSVRC LOC_synset_mapping.txt  LOC_val_solution.csv imagenet_object_localization_patched2019.tar.gz
    >>> LOC_sample_submission.csv  LOC_train_solution.csv  imagenet-object-localization-challenge.zip
    ```
    """
    root = roots['imagenet']
    if split == 'train':
        file = FileBranch(root).branch('ILSVRC', 'ImageSets', 'CLS-LOC').file('train_cls.txt')
        train_root = os.path.join(root, 'ILSVRC/Data/CLS-LOC/train')
        with open(file) as r:
            lines = r.readlines()
            imgs = [line.split(' ')[0] for line in lines]
            name_cls_map = {name: i for i, name in enumerate(sorted(set([i.split('/')[0] for i in imgs])))}
            xs = [os.path.join(train_root, f'{i}.JPEG') for i in imgs]
            ys = [name_cls_map[i.split('/')[0]] for i in imgs]
    else:
        file = FileBranch(root).file('LOC_val_solution.csv')
        val_root = os.path.join(root, 'ILSVRC/Data/CLS-LOC/val')

        with open(file) as r:
            r.readline()
            lines = r.readlines()
            lines = [line.split(',') for line in lines]
            lines = [[img, res.split(' ')[0]] for img, res in lines]

            name_cls_map = {name: i for i, name in enumerate(sorted(set([i[1] for i in lines])))}
            xs = [os.path.join(val_root, f'{img}.JPEG') for img, _ in lines]
            ys = [name_cls_map[res] for _, res in lines]

    return llist(xs), llist(ys)
