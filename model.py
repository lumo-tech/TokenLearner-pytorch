import torch
from torch import nn


class TokenLearnerModule(nn.Module):
    def __init__(self,
                 img_size,
                 num_tokens: int = 8,
                 sum_pooling: bool = True,
                 v11=False,
                 dropout=0.,
                 *args, **kwargs):
        self.sum_pooling = sum_pooling
        self.v11 = v11

        self.ln = nn.LayerNorm(img_size, img_size)  # [bs, c, img_size, img_size]

        if self.v11:
            self.convs = nn.Sequential(
                *[
                    nn.Conv2d(3, num_tokens,
                              kernel_size=(1, 1), stride=(1, 1),
                              padding=0,  # = SAME
                              groups=8,
                              bias=False) for _ in range(3)
                ],
                nn.Conv2d(num_tokens, num_tokens,
                          kernel_size=(1, 1), stride=(1, 1),
                          padding=0,  # = SAME
                          bias=False),
            )
            self.conv2 = nn.Conv2d(3, num_tokens,
                                   kernel_size=(1, 1), stride=(1, 1),
                                   padding=0,  # = SAME
                                   groups=8,  # feature_group_count
                                   bias=False)
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.convs = nn.Sequential(
                *[
                    nn.Conv2d(3, num_tokens,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=1,  # = SAME, (3-1)/2 = 1
                              bias=False) for _ in range(3)
                ],
                nn.GELU(),
                nn.Conv2d(num_tokens, num_tokens,
                          kernel_size=(3, 3), stride=(1, 1),
                          padding=1,  # = SAME, (3-1)/2 = 1
                          bias=False)
            )

    def forward(self, x):
        b, c, h, w = x.shape

        selected = self.ln(x)  # [bs, c, h, w].

        selected = self.convs(selected)  # [bs, n_token, h, w].

        if self.v11:
            selected = torch.reshape(selected, (b, h * w, -1))  # [bs, n_token, h*w].
            selected = torch.softmax(selected, dim=-1)

            feature = self.conv2(x)  # [bs, channels, h, w].
            feature = torch.reshape(feature, (feature.shape[0], h * w, -1))  # [bs, h*w, c].

            feature = torch.bmm(feature, selected)  # einsum('...si,...id->...sd')
            feature = self.dropout(feature)
        else:
            selected = torch.reshape(selected, (selected.shape[0], -1, selected.shape[-1]))  # [bs, h*w, n_token].
            selected = torch.transpose(selected, 1, 2)  # [bs, n_token, h*w].
            selected = torch.sigmoid(selected).unsqueeze(-1)  # [bs, n_token, h*w, 1].

            feature = torch.reshape(x, (x.shape[0], 1, -1, x.shape[-1]))  # [bs, 1, h*w, c].

            feature = feature * selected
            if self.sum_pooling:
                feature = feature.sum(dim=2)
            else:
                feature = feature.mean(dim=2)
        return feature


class TokenLearnerModuleMixer(nn.Module):
    """
    TokenLearner module using the MLPMixer block instead of conv layers..

    Not used.
    """


class TokenFuser(nn.Module):
    def __init__(self,
                 img_size,
                 num_toknes,
                 hidden_size=768,
                 norm: bool = True,
                 dropout: float = 0):
        self.ln = nn.LayerNorm(hidden_size) if norm else nn.Identity()

        self.linaer = nn.Linear(img_size ** 2, num_toknes)

        self.ln2 = nn.LayerNorm(hidden_size)

        self.conv = nn.Sequential(
            nn.Conv2d(3, num_toknes,
                      kernel_size=(1, 1), stride=(1, 1),
                      padding=0,
                      bias=False
                      ),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(p=dropout)

        with torch.no_grad():
            nn.init.zeros_(self.linaer.weight)

    def forward(self, token, ori):
        """

        :param token: Inputs of shape `[bs, n_token, c]`.
        :param ori: Inputs of shape `[bs, h, w, c]`.
        :return:
        """
        token = self.ln(token)
        token = token.transpose(1, 2)  # [bs, c, n_token]

        ori = self.ln2(ori)
        mix = self.conv(ori).unsqueeze(-1)  # [bs, h, w, n_token, 1]

        token = token[:, None, None, :]  # [bs, 1, 1, n_token, c]

        token = token * mix
        token = token.sum(dim=-2)  # [bs, 1, 1, c]
        token = self.dropout(token)
        return token


class EncoderMod(nn.Module):
    def __init__(self,
                 num_layers: int,
                 mlp_dim: int,
                 num_heads: int,
                 dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.1,
                 tokenizer_type: str = 'patch',
                 temporal_dimensions: int = 1,
                 num_tokens: int = 8,
                 tokenlearner_loc: int = 12,
                 use_v11: bool = False,
                 ):
        self.positional_embedding = PositionalEmbs()
        self.dropout = nn.Dropout(dropout_rate)

        self.encoder = nn.Sequential(
            *[
                Encoder1DBlock()
                for i in range(self.num_layres)
            ]
        )
        self.ln = nn.LayerNorm(...)  #
        with torch.no_grad():
            torch.nn.init.normal_(self.positional_embedding, std=0.2)

    def forward(self, inputs):
        pass


class EncoderModFuser(nn.Module):
    def __init__(self,
                 img_size,
                 num_layers: int,
                 mlp_dim: int,
                 num_heads: int,
                 dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.1,
                 tokenizer_type: str = 'patch',
                 temporal_dimensions: int = 1,
                 num_tokens: int = 8,
                 tokenlearner_loc: int = 12,
                 use_v11: bool = False):
        self.positional_embedding = PositionalEmbs()
        self.dropout = nn.Dropout(dropout_rate)

        self.fuser = TokenFuser(...)  # TODO
        self.encoder = nn.Sequential(
            *[
                Encoder1DBlock()
                for i in range(self.num_layres)
            ]
        )
        self.ln = nn.LayerNorm(...)  #
        with torch.no_grad():
            torch.nn.init.normal_(self.positional_embedding, std=0.2)

    def forward(self, inputs):
        pass


class TokenLearnerViT(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 patch_size=16,
                 num_heads=2,
                 num_layers=1,
                 mlp_dim=32,
                 dropout_rate=0.,
                 attention_dropout_rate=0.,
                 hidden_size=16,
                 classifier='gap',
                 data_dtype_str='float32',
                 tokenizer=None,
                 dropout=0.1,
                 tokenfuse=False,
                 token_type='patch',
                 representation_size=None,
                 ):
        fh, fw = patch_size, patch_size

        self.token_type = token_type
        self.conv = nn.Conv2d(
            hidden_size, hidden_size, (fh, fw),
            stride=(fh, fw),
            padding='VALID')
        if tokenfuse:
            self.encoder = EncoderModFuser()
        else:
            self.encoder = EncoderMod()

        self.group_fn = {'gap': torch.mean,
                         'gmp': torch.max,
                         'gsp': torch.sum}[classifier]

        self.linaer = (
            nn.Sequential(
                nn.Linear(hidden_size, representation_size),
                nn.Tanh()
            )
            if representation_size is not None
            else nn.Identity()
        )
        self.linear2 = nn.Linear(
            in_features=hidden_size if representation_size is None else representation_size,
            out_features=num_classes
        )
        with torch.no_grad():
            torch.nn.init.zeros_(self.linear2.weight.data)

    def forward(self, x):
        if self.token_type == 'patch':
            pass
        elif self.token_type == 'dynamic':
            pass
        elif self.token_type == 'video':
            pass
        else:
            raise NotImplementedError('Unknown tokenizer type')

        output = self.conv(x)
        output = self.encoder(output)

        output = self.group_fn(x, dim=1)

        output = self.linaer(output)
        output = self.linaer2(output)
        return output


class TokenLearnerViTRepresentation(nn.Module):
    pass
