#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def mse_masked(img1, img2, mask):
    return ((mask * (img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr_masked(img1, img2, mask):
    mse = ((mask * (img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def shrink_bool_mask(mask: torch.Tensor, iterations: int = 1, kernel_size: int = 3):
    """
    Shrinks a given mask (reduces the number of True pixels) starting
    from the borders of the mask by repeatedly applying min pooling to an image
    `iterations` many times. This is useful to make the system more robust
    to inaccuracies or missed pixels in the masks.

    Taken from our framework to help masking the SSIM loss
    """
    if not kernel_size % 2 == 1:
        raise ValueError("kernel_size for shrink_bool_mask() has to be an odd number!")

    mask_inv = torch.logical_not(mask).unsqueeze(0)

    # Padding to keep the spatial shape the same
    padding = int((kernel_size - 1) / 2)

    # TODO figure out a better way instead of using stride 1 to keep spatial dims the same
    for i in range(iterations):
        mask_inv = F.max_pool2d(mask_inv.float(), kernel_size=kernel_size, stride=1, padding=padding)

        mask_inv = mask_inv.bool()

    # Invert again to get original mask
    mask = torch.logical_not(mask_inv).squeeze()

    return mask