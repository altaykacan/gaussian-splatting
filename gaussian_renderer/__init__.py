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

import math

import torch
import torch.nn.functional as F

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import build_rotation

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,
         return_depth = False, return_normal = False, return_opacity = False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!

    The flags to return the rendered depth, normal maps, and the opacity of the
    gaussians are heavily inspired from the GausianPro implementation:
    https://github.com/kcheng1021/GaussianPro/tree/main
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad() # normally only non-leaf nodes have their gradients stored when .backward() is called
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz # 3D coordinates of the means of the gaussians (centers)
    means2D = screenspace_points # 2D coordinates of the gaussian centers, this python variable only holds gradients, actual values are in the CUDA code
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling # [num_gaussians, 3]
        rotations = pc.get_rotation # [num_gaussians, 4], unit quaternions

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features # rasterizer uses the spherical harmonics to get the color
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return_dict =  {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii
         }

    # Taken from GaussianPro, viewpoint_camera.world_view_transform is the transpose of the 4x4 W2C matrix (T_CW)
    if return_depth:
        projvect1 = viewpoint_camera.world_view_transform[:,2][:3].detach() # third column of the rotation matrix of C2W (corresponds to the z axis of the camera coordinate)
        projvect2 = viewpoint_camera.world_view_transform[:,2][-1].detach() # third component of the translation vector of W2C
        means3D_depth = (means3D * projvect1.unsqueeze(0)).sum(dim=-1,keepdim=True) + projvect2 # first term is the scalar product (orthogonal projection) of the 3D gaussian center to the z axis of the camera (depth)
        means3D_depth = means3D_depth.repeat(1,3)
        render_depth, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = means3D_depth, # hacky way of using the depth values instead of the colors, can use default gaussian rasterizer like this
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        render_depth = render_depth.mean(dim=0) # no need to take the mean, the dummy channel dimension already has the same values
        return_dict.update({'render_depth': render_depth})

    if return_normal:
        rotations_mat = build_rotation(rotations) # [num_points, 3, 3], converts quaternions to rotation matrix
        scales = pc.get_scaling # [num_points, 3]
        min_scales = torch.argmin(scales, dim=1) # [num_points], each entry has the index of the smallest dimension
        indices = torch.arange(min_scales.shape[0]) # [num_points]
        normal = rotations_mat[indices, :, min_scales] # take the corresponding columns of the rotation matrices for each 3D gaussian

        # convert normal direction to the camera; calculate the normal in the camera coordinate
        view_dir = means3D - viewpoint_camera.camera_center # [num_points, 3]

        # The expression ((((view_dir * normal).sum(dim=-1) < 0) * 1 - 0.5) * 2)[..., None] is used to multiply the normals by +1 or -1 depending on if they are parallel or antiparallel with the viewing direction
        normal   = normal * ((((view_dir * normal).sum(dim=-1) < 0) * 1 - 0.5) * 2)[...,None]

        R_w2c = torch.tensor(viewpoint_camera.R.T).cuda().to(torch.float32) # cameras save C2W rotation in their R attribute
        normal = (R_w2c @ normal.transpose(0, 1)).transpose(0, 1) # [num_points, 3], normals in camera coordinate

        render_normal, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = normal,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        render_normal = F.normalize(render_normal, dim = 0) # [3, H, W]
        return_dict.update({'render_normal': render_normal})

    if return_opacity:
        density = torch.ones_like(means3D)

        render_opacity, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = density,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        return_dict.update({'render_opacity': render_opacity.mean(dim=0)})

    return return_dict
