from dataclasses import dataclass
from typing import List, Optional

import torch
from lumo import BaseParams
from torch import nn


class ModelParams(BaseParams):

    def __init__(self):
        super().__init__()
        self.model = self.choice(
            'tokenlearner',
            'vit',
        )


@dataclass()
class ViTOutputs:
    feature_map: Optional[torch.Tensor] = None
    feature: Optional[torch.Tensor] = None
    hidden_states: Optional[List[torch.Tensor]] = None
    last_hidden_state: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class MemoryBank(torch.Tensor):
    def __new__(cls, queue_size=65535, feature_dim=128):
        data = torch.rand(queue_size, feature_dim)
        self = torch.Tensor._make_subclass(cls, data, False)
        self.queue_size = queue_size
        self.cursor = 0
        self.detach_()
        return self

    def to(self, *args, **kwargs):
        ncls = super(MemoryBank, self).to(*args, **kwargs)
        ncls.queue_size = self.queue_size
        ncls.cursor = self.cursor
        return ncls

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data.clone(memory_format=torch.preserve_format), self.requires_grad)
            result.cursor = self.cursor
            result.queue_size = self.queue_size
            memo[id(self)] = result
            return result

    def push(self, item: torch.Tensor):
        assert item.ndim == 2 and item.shape[1:] == self.shape[1:], f'ndim: {item.ndim} | shape: {item.shape}'
        with torch.no_grad():
            item = item.to(self.data.device)
            isize = len(item)
            if self.cursor + isize > self.queue_size:
                right = self.queue_size - self.cursor
                left = isize - right
                self.data[self.cursor:] = item[:right]
                self.data[:left] = item[right:]
            else:
                self.data[self.cursor:self.cursor + len(item)] = item
            self.cursor = (self.cursor + len(item)) % self.queue_size

    def tensor(self):
        return torch.Tensor._make_subclass(torch.Tensor, self.data, False)

    def clone(self, *args, **kwargs):
        ncls = super(MemoryBank, self).clone(*args, **kwargs)
        ncls.queue_size = self.queue_size
        ncls.cursor = self.cursor
        return ncls

    def half(self, memory_format=None):
        return self.to(torch.float16)


class LongTensorMemoryBank(torch.Tensor):
    def __new__(cls, queue_size=65535):
        data = torch.zeros(queue_size, dtype=torch.long)
        self = torch.Tensor._make_subclass(cls, data, False)
        self.queue_size = queue_size
        self.cursor = 0
        self.detach_()
        return self

    def to(self, *args, **kwargs):
        ncls = super(LongTensorMemoryBank, self).to(*args, **kwargs)
        ncls.queue_size = self.queue_size
        ncls.cursor = self.cursor
        return ncls

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data.clone(memory_format=torch.preserve_format), self.requires_grad)
            result.cursor = self.cursor
            result.queue_size = self.queue_size
            memo[id(self)] = result
            return result

    def push(self, item: torch.Tensor):
        with torch.no_grad():
            item = item.to(self.device)
            isize = len(item)
            if self.cursor + isize > self.queue_size:
                right = self.queue_size - self.cursor
                left = isize - right
                self.data[self.cursor:] = item[:right]
                self.data[:left] = item[right:]
            else:
                self.data[self.cursor:self.cursor + len(item)] = item
            self.cursor = (self.cursor + len(item)) % self.queue_size

    def tensor(self):
        return torch.Tensor._make_subclass(torch.Tensor, self.data, False)

    def clone(self, *args, **kwargs):
        ncls = super(LongTensorMemoryBank, self).clone(*args, **kwargs)
        ncls.queue_size = self.queue_size
        ncls.cursor = self.cursor
        return ncls

    def half(self, memory_format=None):
        return self.to(torch.float16)


def pick_model_name(model_name) -> nn.Module:
    from . import vit_tokenlearner, vit

    if model_name in {'vit'}:
        model = vit.ViT(
            image_size=96,
            patch_size=16,
            dim=768,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
    elif model_name in {'tokenlearner'}:
        model = vit_tokenlearner.ViT(
            image_size=96,
            num_tokens=8,
            fuse=False,
            v11=True,
            tokenlearner_loc=3,
            patch_size=16,
            hidden_size=768,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

    return model
