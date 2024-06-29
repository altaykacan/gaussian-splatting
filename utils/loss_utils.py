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
from torch.autograd import Variable
from math import exp
from utils.image_utils import shrink_bool_mask

def constant_opacity_loss(opacities: torch.Tensor, target: float):
    """
    Loss term that penalizes any deviation from a target opacity
    for the (visible) gaussians.
    """
    # TODO can we mask this in a meaningful way? The opacities is an unordered list of the gaussian opacities
    return torch.mean(torch.abs(opacities - target))


def opacity_entropy_loss(opacities: torch.Tensor):
    """
    Loss term that minimizes the entropies of the learnable opacity
    distributions of the (visible) gaussians. This is different from
    the alpha entropy regularization as this loss uses only the opacity,
    not the evaluated 2D gaussian.
    """
    return torch.mean(-opacities * torch.log(opacities))


def disk_loss(scales: torch.Tensor):
    """
    Loss term that penalizes each (visible) gaussian if it's largest two scale
    values are different from one another and if the smallest scale value is large

    Scales has shape (num_visible_gaussians, 3)
    """
    top_two, _ = torch.topk(scales, k=2, dim=1)
    scale_min, _ = torch.min(scales, dim=1)

    top_difference = top_two[:, 0] - top_two[:, 1]

    return torch.mean(top_difference**2 + scale_min**2)


def total_variation_loss(depth: torch.Tensor, mask: torch.Tensor=None):
    """
    Computes total variation loss intended to make the depth renderings smoother.
    Implementation inspired by DN-Splatter:
    https://github.com/maturk/dn-splatter/blob/main/dn_splatter/losses.py#L269

    Args:
        depth: Tensor representing the depth renderings of the
               model of shape (H, W)
        mask: Boolean tensor with shape (H,W), has True values for where the loss
              should be computed at
    """
    if mask is None:
        mask = torch.ones_like(depth)

    depth = depth * mask

    # These do not have shape [H, W] shape so we mask the depth first
    h_diff = depth[..., :, :-1] - depth[..., :, 1:] # TODO figure out whether this will propagate to the masked regions or from the masked regions?
    w_diff = depth[..., :-1, :] - depth[..., 1:, :]

    return torch.mean(torch.abs(h_diff)) + torch.mean(torch.abs(w_diff))

def log_depth_loss(network_output, gt, mask=None):
    """
    Loss term for depth regularization using a logarithm
    around the L1 loss.

    Inspired by DN-Splatter's LogL1 loss:
    https://github.com/maturk/dn-splatter/blob/main/dn_splatter/losses.py#L161
    """
    if mask is None:
        mask = torch.ones_like(network_output)

    return torch.mean(torch.log(1 + torch.abs(network_output - gt)) * mask)


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l1_loss_mask(network_output, gt, mask):
    """
    Custom l1 loss with masking using a boolean tensor. Computes the l1 loss
    for True elements in the mask and computes the mean accross all pixels
    to equally weight them.
    """
    return torch.mean(torch.abs((network_output - gt)) * mask)


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim_mask(img1, img2, mask, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    # Assumes window size will be odd, basically grows the ignored regions
    mask = shrink_bool_mask(mask, iterations=1, kernel_size=window_size)

    # # mask has the pixels True where we should compute the loss
    # mask = torch.logical_not(mask)

    # # If gt is an rgb image with 3 channels, we need to expand our binary mask to match it
    # if len(img2.shape) == 3 and len(mask.shape) == 2:
    #     mask = mask[None, :, :].expand(3, -1, -1)

    # img2[mask] = img1.detach()[mask] #trying alternative strategy for masking, replacing gt with prediction

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_mask(img1, img2, mask, window, window_size, channel, size_average)

def _ssim_mask(img1, img2, mask, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        # ssim_map has the same size as img1 and img2 so we can directly use the resized map
        # return torch.sum(ssim_map * mask) / torch.sum(mask)
        return torch.mean(ssim_map * mask)
        # return torch.mean(ssim_map)
    else:
        return ssim_map.mean(1).mean(1).mean(1)

