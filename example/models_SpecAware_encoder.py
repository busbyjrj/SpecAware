# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from timm.layers import trunc_normal_
from HyperEmbedding import HyperEmbedding
from util.pos_embed import get_2d_sincos_pos_embed_with_resolution



class MaskedHSIAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.,
        norm_layer=nn.LayerNorm,
        cls_embed=True,
        drop_path_rate=0.,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.cls_embed = cls_embed
        self.embed_dim = embed_dim
    
    
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = HyperEmbedding(
            img_size=img_size,
            wv_planes=128,
            kernel_size=patch_size,
            embed_dim=embed_dim
        )

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(embed_dim,
                  num_heads,
                  mlp_ratio,
                  qkv_bias=True,
                  drop_path=dpr[i],
                  norm_layer=norm_layer) for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        # HyperMAE encoder END--------------------------------------------------------------------------



    def forward(self, imgs, GSD=None, wavelength=None, fwhm=None, sensor_name=None, data_level=None):
        B, C, H, W = imgs.shape

        if GSD is None:
            GSD = torch.tensor([10], device=imgs.device) * torch.ones(imgs.shape[0], device=imgs.device)
        if len(GSD.shape) == 2:
            GSD = GSD.squeeze(1)
        if len(wavelength.shape) == 1:
            wavelength = wavelength.unsqueeze(0).repeat(B, 1)
        if len(fwhm.shape) == 1:
            fwhm = fwhm.unsqueeze(0).repeat(B, 1)

        x, _ = self.patch_embed(imgs, wavelength, fwhm, sensor_name, data_level)

        # get 2d pos embed
        assert len(GSD.shape) == 1, f"GSD shape: {GSD.shape}"
        pos_embed = get_2d_sincos_pos_embed_with_resolution(
            self.embed_dim,
            H // self.patch_size,
            GSD.float(),
            cls_token=True)

        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :].type_as(x).to(
            x.device).clone().detach()

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token + pos_embed[:, :1, :].type_as(
                x).to(x.device).clone().detach()
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        features = []
        feature_idx = [2, 5, 8, 11]
        # feature_idx = [5, 11, 17, 23]
        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)

            if i == len(self.blocks) - 1:
                x = self.norm(x)

            if i in feature_idx:
                if self.cls_embed:
                    out = x[:, 1:, :] 
                else:
                    out = x
                B, L, D = out.shape
                H = W = int(L ** 0.5)
                out = out.permute(0, 2, 1).reshape(B, D, H, W)
                features.append(out)

        return features, x



def mae_vit_base_patch8_hsi(**kwargs):
    model = MaskedHSIAutoencoderViT(embed_dim=768,
                                    depth=12,
                                    num_heads=12,
                                    mlp_ratio=4,
                                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                    **kwargs)
    return model


mae_vit_base_patch8 = mae_vit_base_patch8_hsi
