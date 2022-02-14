from lumo import llist
from PIL import Image
from lumo.proc.path import cache_dir
from torchvision.datasets import STL10
import numpy as np


def stl10(split='train'):
    """
    96x96x3
    """
    try:
        dataset = STL10(root=cache_dir(), split=split, download=False)
    except:
        dataset = STL10(root=cache_dir(), split=split, download=True)
    xs = llist(Image.fromarray(np.transpose(i, (1, 2, 0))) for i in dataset.data)
    ys = llist(int(i) for i in dataset.labels)

    return xs, ys
