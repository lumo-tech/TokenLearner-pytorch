from torch import nn

conv = nn.Conv2d(3, 8, (3, 3), (1, 1), padding=1)

ln = nn.LayerNorm(32, 32)
import torch

x = torch.rand(1, 3, 32, 32)

y = ln(x)
print(y.shape)
