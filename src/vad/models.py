"""Reconstruction models: a spatio-temporal Convolutional Autoencoder (CAE) and a
masked Vision-Transformer autoencoder (ViT-AE). Both take a clip of stacked frames
(and optional motion channels) shaped (B, C, H, W) and return a reconstruction."""
from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin, cout, n=2):
    layers = []
    for i in range(n):
        layers += [nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True)]
    return nn.Sequential(*layers)


class ConvAutoencoder(nn.Module):
    """Hasan et al. (2016)-style CAE: frames stacked along channels, 3x downsampling
    to a compact bottleneck, then symmetric transposed-conv decoder (no skips, so the
    model cannot copy its input and must learn the normal manifold)."""

    def __init__(self, in_ch: int, base: int = 24, latent: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            _block(in_ch, base, n=1), nn.MaxPool2d(2),  # one conv at full resolution keeps CPU training fast
            _block(base, base * 2), nn.MaxPool2d(2),
            _block(base * 2, base * 4), nn.MaxPool2d(2),
            nn.Conv2d(base * 4, latent, 3, padding=1), nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent, base * 4, 2, stride=2), nn.LeakyReLU(0.2, inplace=True), _block(base * 4, base * 4),
            nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2), nn.LeakyReLU(0.2, inplace=True), _block(base * 2, base * 2),
            nn.ConvTranspose2d(base * 2, base, 2, stride=2), nn.LeakyReLU(0.2, inplace=True), _block(base, base, n=1),
            nn.Conv2d(base, in_ch, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class MaskedViTAutoencoder(nn.Module):
    """ViT autoencoder trained as masked patch inpainting (MAE-style).

    Training: a random 50% of patch tokens are replaced by a learned mask token and
    the model must reconstruct them from context. Inference: two complementary
    checkerboard masks are applied and each patch is predicted while hidden, so an
    unusual object can't be 'copied' through and shows up as high error."""

    def __init__(self, in_ch: int, img_size=(128, 192), patch: int = 16, dim: int = 192,
                 depth: int = 4, heads: int = 4, mask_ratio: float = 0.5):
        super().__init__()
        self.patch, self.in_ch, self.mask_ratio = patch, in_ch, mask_ratio
        self.gh, self.gw = img_size[0] // patch, img_size[1] // patch
        n = self.gh * self.gw
        self.embed = nn.Linear(in_ch * patch * patch, dim)
        self.pos = nn.Parameter(torch.zeros(1, n, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, in_ch * patch * patch)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def patchify(self, x):
        b, c, h, w = x.shape
        p = self.patch
        x = x.reshape(b, c, self.gh, p, self.gw, p).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(b, self.gh * self.gw, c * p * p)

    def unpatchify(self, t):
        b = t.shape[0]
        p, c = self.patch, self.in_ch
        t = t.reshape(b, self.gh, self.gw, c, p, p).permute(0, 3, 1, 4, 2, 5)
        return t.reshape(b, c, self.gh * p, self.gw * p)

    def _run(self, x, mask):  # mask: (B, N) bool, True = hidden
        tok = self.embed(self.patchify(x))
        tok = torch.where(mask[..., None], self.mask_token.expand_as(tok), tok) + self.pos
        return torch.sigmoid(self.head(self.norm(self.encoder(tok))))

    def checkerboard(self, b, device):
        ii, jj = torch.meshgrid(torch.arange(self.gh), torch.arange(self.gw), indexing="ij")
        m = ((ii + jj) % 2 == 0).flatten().to(device)
        return m.expand(b, -1), (~m).expand(b, -1)

    def forward(self, x):
        b = x.shape[0]
        if self.training:
            n = self.gh * self.gw
            mask = torch.rand(b, n, device=x.device).argsort(1) < int(n * self.mask_ratio)
            out = self._run(x, mask)
            return self.unpatchify(out), mask
        m1, m2 = self.checkerboard(b, x.device)
        o1, o2 = self._run(x, m1), self._run(x, m2)
        out = torch.where(m1[..., None], o1, o2)
        return self.unpatchify(out)


def build_model(name: str, in_ch: int, img_size) -> nn.Module:
    if name == "cae":
        return ConvAutoencoder(in_ch)
    if name == "vit":
        return MaskedViTAutoencoder(in_ch, img_size)
    raise ValueError(f"unknown model {name}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
