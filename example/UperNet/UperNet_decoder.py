# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : UperNet.py
# Time       : 2024/12/22 21:23
# Author     : Renjie Ji
# Email      : busbyjrj@gmail.com
# Description:
# Reference  : https://github.com/sherif-med/segmentation_models.pytorch/blob/2cc41bdb3f306e14084bcd0ef92914b7d5c613e7/segmentation_models_pytorch/decoders/upernet/decoder.py#L73
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

InPlaceABN = None


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class PSPModule(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            sizes=(1, 2, 3, 6),
            use_batchnorm=True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    Conv2dReLU(
                        in_channels,
                        in_channels // len(sizes),
                        kernel_size=1,
                        use_batchnorm=use_batchnorm,
                    ),
                )
                for size in sizes
            ]
        )
        self.out_conv = Conv2dReLU(
            in_channels=in_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x):
        _, _, height, width = x.shape
        out = [x] + [
            F.interpolate(
                block(x), size=(height, width), mode="bilinear", align_corners=False
            )
            for block in self.blocks
        ]
        out = self.out_conv(torch.cat(out, dim=1))
        return out


class FPNBlock(nn.Module):
    def __init__(self, skip_channels, pyramid_channels, use_batchnorm=True):
        super().__init__()
        self.skip_conv = (
            Conv2dReLU(
                skip_channels,
                pyramid_channels,
                kernel_size=1,
                use_batchnorm=use_batchnorm,
            )
            if skip_channels != 0
            else nn.Identity()
        )

    def forward(self, x, skip):
        _, channels, height, width = skip.shape
        x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        if channels != 0:
            skip = self.skip_conv(skip)
            x = x + skip
        return x


class UPerNetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            encoder_depth=4,
            pyramid_channels=256,
            segmentation_channels=64,
            target_upsample=2,
    ):
        super().__init__()
        self.target_upsample = target_upsample

        if encoder_depth < 3:
            raise ValueError(
                "Encoder depth for UPerNet decoder cannot be less than 3, got {}.".format(
                    encoder_depth
                )
            )

        encoder_channels = encoder_channels[::-1]

        # PSP Module
        self.psp = PSPModule(
            in_channels=encoder_channels[0],
            out_channels=pyramid_channels,
            sizes=(1, 2, 3, 6),
            use_batchnorm=True,
        )

        # FPN Module
        self.fpn_stages = nn.ModuleList(
            [FPNBlock(ch, pyramid_channels) for ch in encoder_channels[1:]]
        )

        self.fpn_bottleneck = Conv2dReLU(
            # in_channels=(len(encoder_channels) - 1) * pyramid_channels,
            in_channels=(len(encoder_channels) * pyramid_channels),
            out_channels=segmentation_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )

    def forward(self, *features):
        output_size = features[0].shape[2:]
        target_size = [size * self.target_upsample for size in output_size]

        # features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        psp_out = self.psp(features[0])

        fpn_features = [psp_out]
        for feature, stage in zip(features[1:], self.fpn_stages):
            fpn_feature = stage(fpn_features[-1], feature)
            fpn_features.append(fpn_feature)

        # Resize all FPN features to 1/4 of the original resolution.
        resized_fpn_features = []
        for feature in fpn_features:
            resized_feature = F.interpolate(
                feature, size=target_size, mode="bilinear", align_corners=False
            )
            resized_fpn_features.append(resized_feature)

        output = self.fpn_bottleneck(torch.cat(resized_fpn_features, dim=1))

        return output
