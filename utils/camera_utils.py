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

from scene.cameras import Camera
import torch
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution)
        )
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]  # only take the first
    loaded_mask = None

    if (
        resized_image_rgb.shape[1] == 4
    ):  # if we have a png with 4 channels we save alpha mask, I think this is a bug [C, H, W] is the shape so we should check index 0 not 1 -altay
        loaded_mask = resized_image_rgb[3:4, ...]

    if cam_info.mask is not None:
        mask = cam_info.mask
        mask = torch.from_numpy(mask)
        # TODO add code to resize/adjust the mask
    else:
        mask = None

    if cam_info.gt_depth is not None:
        gt_depth = cam_info.gt_depth
        gt_depth = torch.from_numpy(gt_depth)
        # TODO add code to resize/adjust the depth
    else:
        gt_depth = None

    if cam_info.gt_normal is not None:
        gt_normal = cam_info.gt_normal
        gt_normal = torch.from_numpy(gt_normal)
        # TODO add code to resize/adjust the normal maps, make sure that the values are normalized
    else:
        gt_normal = None

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
        mask=mask,
        gt_depth=gt_depth,
        gt_normal=gt_normal,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry


def perturb_viewpoint(
    viewpoint: Camera,
    scene_extent: float,
    displacement: float = 1.0,
    scene_extent_percentage: float = 0.01,
    rot_angle: float = 45,
):
    uid = viewpoint.uid
    colmap_id = viewpoint.colmap_id
    R = viewpoint.R  # This is the rotation of W2C
    T = viewpoint.T  # This is the translation of C2W
    FoVx = viewpoint.FoVx
    FoVy = viewpoint.FoVy
    image = viewpoint.original_image  # not used for perturbed viewpoints
    image_name = viewpoint.image_name
    data_device = viewpoint.data_device

    rot_angle = np.pi / 180 * rot_angle # convert to radians

    perturbed_viewpoints = {}

    # x displacement
    delta_x = (
        scene_extent * scene_extent_percentage * np.array([displacement, 0.0, 0.0])
    )

    perturbed_viewpoints["x_positive"] = Camera(
        colmap_id=colmap_id,
        R=R,
        T=T + delta_x,
        FoVx=FoVx,
        FoVy=FoVy,
        image=image,
        gt_alpha_mask=None,
        image_name=image_name + "_x_positive",
        uid=uid,
        data_device=data_device,
        mask=None,
        gt_depth=None,
        gt_normal=None,
    )

    perturbed_viewpoints["x_negative"] = Camera(
        colmap_id=colmap_id,
        R=R,
        T=T - delta_x,
        FoVx=FoVx,
        FoVy=FoVy,
        image=image,
        gt_alpha_mask=None,
        image_name=image_name + "_x_negative",
        uid=uid,
        data_device=data_device,
        mask=None,
        gt_depth=None,
        gt_normal=None,
    )

    # y displacement
    delta_y = (
        scene_extent * scene_extent_percentage * np.array([0.0, displacement, 0.0])
    )

    perturbed_viewpoints["y_positive"] = Camera(
        colmap_id=colmap_id,
        R=R,
        T=T + delta_y,
        FoVx=FoVx,
        FoVy=FoVy,
        image=image,
        gt_alpha_mask=None,
        image_name=image_name + "_y_positive",
        uid=uid,
        data_device=data_device,
        mask=None,
        gt_depth=None,
        gt_normal=None,
    )

    perturbed_viewpoints["y_negative"] = Camera(
        colmap_id=colmap_id,
        R=R,
        T=T - delta_y,
        FoVx=FoVx,
        FoVy=FoVy,
        image=image,
        gt_alpha_mask=None,
        image_name=image_name + "_y_negative",
        uid=uid,
        data_device=data_device,
        mask=None,
        gt_depth=None,
        gt_normal=None,
    )

    # z displacement
    delta_z = (
        scene_extent * scene_extent_percentage * np.array([0.0, 0.0, displacement])
    )

    perturbed_viewpoints["z_positive"] = Camera(
        colmap_id=colmap_id,
        R=R,
        T=T + delta_z,
        FoVx=FoVx,
        FoVy=FoVy,
        image=image,
        gt_alpha_mask=None,
        image_name=image_name + "_z_positive",
        uid=uid,
        data_device=data_device,
        mask=None,
        gt_depth=None,
        gt_normal=None,
    )

    perturbed_viewpoints["z_negative"] = Camera(
        colmap_id=colmap_id,
        R=R,
        T=T - delta_z,
        FoVx=FoVx,
        FoVy=FoVy,
        image=image,
        gt_alpha_mask=None,
        image_name=image_name + "z_negative",
        uid=uid,
        data_device=data_device,
        mask=None,
        gt_depth=None,
        gt_normal=None,
    )

    # Combined displacement
    delta = (
        scene_extent
        * scene_extent_percentage
        * np.array([displacement, displacement, displacement])
    )

    perturbed_viewpoints["disp_positive"] = Camera(
        colmap_id=uid,
        R=R,
        T=T + delta,
        FoVx=FoVx,
        FoVy=FoVy,
        image=image,
        gt_alpha_mask=None,
        image_name=image_name + "_disp_positive",
        uid=uid,
        data_device=data_device,
        mask=None,
        gt_depth=None,
        gt_normal=None,
    )

    perturbed_viewpoints["disp_negative"] = Camera(
        colmap_id=colmap_id,
        R=R,
        T=T - delta,
        FoVx=FoVx,
        FoVy=FoVy,
        image=image,
        gt_alpha_mask=None,
        image_name=image_name + "_disp_negative",
        uid=uid,
        data_device=data_device,
        mask=None,
        gt_depth=None,
        gt_normal=None,
    )

    # Rotation around y
    rot = np.array([[np.cos(rot_angle),  0.0, np.sin(rot_angle)],
                    [0.0,                1.0,              0.0 ],
                    [-np.sin(rot_angle), 0.0, np.cos(rot_angle)]])

    # R stores the rotation of W2C so we need to inverse it to get the rotation transform
    # for easy composition. We also need to invert the combined rotations to get
    # it back to W2C format
    R_positive = (rot @ R.T).T
    R_negative = (rot.T @ R.T).T

    perturbed_viewpoints["rot_positive"] = Camera(
        colmap_id=colmap_id,
        R=R_positive,
        T=T,
        FoVx=FoVx,
        FoVy=FoVy,
        image=image,
        gt_alpha_mask=None,
        image_name=image_name + "_rot_positive",
        uid=uid,
        data_device=data_device,
        mask=None,
        gt_depth=None,
        gt_normal=None,
    )

    perturbed_viewpoints["rot_negative"] = Camera(
        colmap_id=colmap_id,
        R=R_negative,
        T=T,
        FoVx=FoVx,
        FoVy=FoVy,
        image=image,
        gt_alpha_mask=None,
        image_name=image_name + "_rot_negative",
        uid=uid,
        data_device=data_device,
        mask=None,
        gt_depth=None,
        gt_normal=None,
    )

    # Combined displacement and rotation
    perturbed_viewpoints["combined_positive"] = Camera(
        colmap_id=colmap_id,
        R=R_positive,
        T=T + delta,
        FoVx=FoVx,
        FoVy=FoVy,
        image=image,
        gt_alpha_mask=None,
        image_name=image_name + "_combined_positive",
        uid=uid,
        data_device=data_device,
        mask=None,
        gt_depth=None,
        gt_normal=None,
    )

    perturbed_viewpoints["combined_negative"] = Camera(
        colmap_id=colmap_id,
        R=R_negative,
        T=T - delta,
        FoVx=FoVx,
        FoVy=FoVy,
        image=image,
        gt_alpha_mask=None,
        image_name=image_name + "_combined_negative",
        uid=uid,
        data_device=data_device,
        mask=None,
        gt_depth=None,
        gt_normal=None,
    )

    return perturbed_viewpoints
