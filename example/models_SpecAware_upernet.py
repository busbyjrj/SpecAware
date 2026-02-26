# https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/decoders/upernet/model.py
# https://github.com/open-mmlab/mmsegmentation/blob/main/configs/_base_/models/upernet_vit-b16_ln_mln.py

from functools import partial
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from UperNet.MultiLevelNeck import MultiLevelNeck
from UperNet.UperNet_decoder import UPerNetDecoder
from models_SpecAware_encoder import MaskedHSIAutoencoderViT


class SegmentationHead(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1
    ):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        # activation = Activation(activation)
        activation = nn.Identity()
        super().__init__(conv2d, upsampling, activation)


class ViTUPerNet(nn.Module):
    """UPerNet is a unified perceptual parsing network for image segmentation.

    Returns:
        ``torch.nn.Module``: **UPerNet**

    .. _UPerNet:
        https://arxiv.org/abs/1807.10221

    """

    def __init__(
            self,
            encoder_depth: int = 4,
            encoder_weights: Optional[str] = None,
            img_size=256,
            decoder_pyramid_channels: int = 256,
            decoder_segmentation_channels: int = 64,
            in_chans: int = 3,
            num_classes: int = 1,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            norm_layer: nn.Module = nn.LayerNorm,
            qkv_bias: bool = True,
            patch_size: int = 8,
            activation: Optional[Union[str, callable]] = None,
            aux_params: Optional[dict] = None,
            seg_upsampling=1,
            drop_path_rate=0.,
            **kwargs: Any,
    ):
        super().__init__()

        encoder_name = "ViTUPerNet"
        
        self.encoder = MaskedHSIAutoencoderViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_path_rate=drop_path_rate,
            **kwargs)

        classes = num_classes
        neck_in_channels = [embed_dim] * 4
        neck_out_channels = decoder_pyramid_channels
        
        encoder_channels = [neck_out_channels] * 4


        self.neck = MultiLevelNeck(
            in_channels=neck_in_channels,
            out_channels=neck_out_channels,
            scales=[4, 2, 1, 0.5],
        )

        self.decoder = UPerNetDecoder(
            encoder_channels=encoder_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            target_upsample=max(1, patch_size // 4),
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_segmentation_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=1,
        )

        self.name = "upernet-{}".format(encoder_name)

    def check_input_shape(self, x):
        h, w = x.shape[-2:]
        output_stride = self.encoder.patch_size
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (
                (h // output_stride + 1) * output_stride
                if h % output_stride != 0
                else h
            )
            new_w = (
                (w // output_stride + 1) * output_stride
                if w % output_stride != 0
                else w
            )
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    @torch.jit.ignore()
    def no_weight_decay(self):
        return {
            "cls_token",
            "pos_embed",
            "dist_token"
        }

    
    def load_pretrained_weights(self, checkpoint_path, old_patch_size=8, strict=False, verbose=False, mode="bilinear"):
        interp_kwargs = dict(mode=mode)
        if mode == 'bilinear':
            interp_kwargs['align_corners'] = False
        elif mode == 'nearest':
            pass

        new_ps = self.encoder.patch_size
        state = torch.load(checkpoint_path, map_location='cpu')
        state = {f"encoder.{k}": v for k, v in state.items()}

        if new_ps != old_patch_size:
            adapted = self._adapt_state_dict(
                state, old_patch_size, new_ps, interp_kwargs, verbose
            )
        else:
            adapted = state

        self.load_state_dict(adapted, strict=strict)

        if verbose:
            print(f"Loaded pretrained weights from {checkpoint_path}. ")

    def _adapt_state_dict(self, state, old_ps, new_ps, interp_kwargs=dict(mode="bilinear"), verbose=False):
        """Adapt state dict with interpolation for patch size mismatch."""
        model_state = self.state_dict()
        adapted = {}

        for key, value in state.items():
            if key not in model_state:
                continue
            if value.shape == model_state[key].shape:
                adapted[key] = value
                continue
            
            # Try interpolation for v_generator weights
            if (old_ps is not None and new_ps is not None and
                'matrix_generator' in key and 'v_generator' in key and
                value.dim() > 0 and value.shape[0] % (old_ps * old_ps) == 0):
                
                try:
                    if key.endswith('.weight') and value.dim() == 2:
                        interpolated = self._resize_patch_weight(value, old_ps, new_ps, interp_kwargs, verbose, key)
                        adapted[key] = interpolated
                        continue
                    elif key.endswith('.bias') and value.dim() == 1:
                        interpolated = self._resize_patch_bias(value, old_ps, new_ps, interp_kwargs, verbose, key)
                        adapted[key] = interpolated
                        continue
                except Exception as e:
                    if verbose:
                        print(f"[interpolate] error {key}: {e}")
            
        return adapted

    def _resize_patch_weight(self, param, old_ps, new_ps,
                            interp_kwargs=dict(mode="bilinear"), verbose=False, key=None) -> torch.Tensor:   
        out_features, in_features = param.shape
        rank = out_features // (old_ps * old_ps)

        # [rank, in, old_ps, old_ps] -> [rank, in, new_ps, new_ps]
        p = param.view(rank, old_ps, old_ps, in_features).permute(0, 3, 1, 2)
        p = F.interpolate(p, size=(new_ps, new_ps), **interp_kwargs)
        result = p.permute(0, 2, 3, 1).reshape(-1, in_features)
        
        if verbose:
            print(f"[interpolate] v_gen weight {key}: {param.shape} -> {result.shape}, ps: {old_ps}→{new_ps})")
        return result

    def _resize_patch_bias(self, param, old_ps, new_ps,
                        interp_kwargs=dict(mode="bilinear"), verbose=False, key=None) -> torch.Tensor:
        rank = param.numel() // (old_ps * old_ps)

        # [rank, 1, old_ps, old_ps] -> [rank, 1, new_ps, new_ps]
        b = param.view(rank, 1, old_ps, old_ps)
        b = F.interpolate(b, size=(new_ps, new_ps), **interp_kwargs)
        result = b.reshape(-1)
        
        if verbose:
            print(f"[interpolate] v_gen bias {key}: {param.shape} -> {result.shape}")
        return result

    def forward(self, x, GSD, wavelength, fwhm, sensor_name, data_level):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        B, C, H, W = x.shape
        # if not torch.jit.is_tracing():
        #     self.check_input_shape(x)

        features, _ = self.encoder(x, GSD, wavelength, fwhm, sensor_name, data_level)
        features = self.neck(features)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        
        return masks


        
if __name__ == "__main__":
    img_size = 256
    patch_size = 8
    old_patch_size = 8
    in_chans = 100
    num_classes = 8

    model = ViTUPerNet(
        img_size=img_size,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=nn.LayerNorm,
        qkv_bias=True,
        patch_size=patch_size,
        seg_upsampling=1,
    )
    model = model.to("cuda")

    ckpt = "./SpecAware_Base_model.pth"
    model.load_pretrained_weights(ckpt, old_patch_size=8, strict=False, mode="bilinear")  # nearest
    
    import numpy as np
    B, C, H, W = 4, in_chans, img_size, img_size
    x = torch.randn(B, C, H, W).to("cuda")
    wavelengths = [np.linspace(400, 2500, C).astype(np.float32) for _ in range(B)]
    wavelengths = np.array(wavelengths)
    wavelengths = torch.from_numpy(wavelengths).to("cuda")
    fwhm = [np.linspace(6.0, 10.0, C).astype(np.float32) for _ in range(B)]
    fwhm = np.array(fwhm)
    fwhm = torch.from_numpy(fwhm).to("cuda")
    sensor_name = ["av3" for _ in range(B)]
    data_level = ["L2" for _ in range(B)]
    x = x.to("cuda")
    GSD = torch.tensor([10.0]).to("cuda")
    GSD = GSD * patch_size / old_patch_size
    
    out = model(x, GSD=GSD, wavelength=wavelengths, fwhm=fwhm, sensor_name=sensor_name, data_level=data_level)
    print("out.shape:", out.shape)
