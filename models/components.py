from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self,
                 feature_dim=128,
                 mid_dim=128,  # hidden_size
                 output_dim=64, with_bn=False, with_leakyrelu=True):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(feature_dim, mid_dim),
            nn.BatchNorm1d(mid_dim) if with_bn else nn.Identity(),
            nn.LeakyReLU(negative_slope=0.1, inplace=True) if with_leakyrelu else nn.ReLU(inplace=True),
            nn.Linear(mid_dim, output_dim),
        )

    def forward(self, feature):
        return self.module(feature)


class MLP2(nn.Module):
    def __init__(self,
                 feature_dim=128,
                 mid_dim=128,
                 output_dim=64, with_bn=False, with_leakyrelu=True):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(feature_dim, mid_dim),
            nn.BatchNorm1d(mid_dim) if with_bn else nn.Identity(),
            nn.LeakyReLU(negative_slope=0.1, inplace=True) if with_leakyrelu else nn.ReLU(inplace=True),
            nn.Linear(mid_dim, output_dim),
            nn.BatchNorm1d(output_dim) if with_bn else nn.Identity(),
        )

    def forward(self, feature):
        return self.module(feature)


class NormMLP(MLP):
    def forward(self, feature):
        return F.normalize(super().forward(feature), p=2, dim=-1)


class ResidualLinear(nn.Module):
    def __init__(self, in_feature, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_feature, in_feature, bias=bias)

    def forward(self, feature):
        out = self.linear(feature)
        out = out + feature
        return out
