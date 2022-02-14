import torch
from models.vit_tokenlearner import ViT

v = ViT(
    image_size=256,
    num_tokens=8,
    fuse=False,
    v11=True,
    tokenlearner_loc=3,
    patch_size=16,
    num_classes=1000,
    hidden_size=768,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

img = torch.randn(1, 3, 256, 256)

preds = v(img)  # (1, 1000)
print(preds.shape)
