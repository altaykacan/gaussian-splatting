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

import os
import sys
from PIL import Image
from typing import NamedTuple, Union
from scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    read_points3D_text,
)
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import torch
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

from scene.densecloud_loader import (
    read_densecloud_extrinsics,
    read_densecloud_extrinsics_colmap,
    read_densecloud_extrinsics_colmap_binary,
    read_densecloud_intrinsics,
)


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array = None  # optional mask for loss computation
    gt_depth: np.array = None  # optional ground truth depth for depth regularization
    gt_normal: np.array = (
        None  # optional ground truth normals for normal regularization
    )


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    scene_scale: float


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center  # avg camera center coordinates in world frame
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(
            dist
        )  # distance of the furthest away camera center from the average center
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(
            C2W[:3, 3:4]
        )  # list of camera center coordinates in world frame

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(
            qvec2rotmat(extr.qvec)
        )  # TODO figure out why do we take the transpose? -altay
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0

    try:
        normals = np.vstack(
            [vertices["nx"], vertices["ny"], vertices["nz"]]
        ).T  # for our case the normals are all 0's -altay
    except Exception as E:
        print(
            f"Encountered exception `{E}` when trying to load normals. Setting all zeros for the normals."
        )
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)  # normals are effectively ignored -altay

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    """
    Reads relevant scene information from the source path `path`.
    Returns a `SceneInfo` object that contains information about the pointcloud,
    train and test views (lists of `CameraInfo` objects), normalization values
    for NeRF normalization, and the path to the `.ply` file representing
    the pointcloud for initializing the gaussian model.
    """
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print(
            "Converting point3d.bin to .ply, will happen only the first time you open the scene."
        )
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)  # returns a simple NamedTuple of numpy arrays
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    """Helper function used for readNerfSyntheticInfo to read in cameras"""
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                1 - norm_data[:, :, 3:4]
            )
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                )
            )

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension
    )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readDenseCloudCameras(
    cam_extrinsics,
    cam_intrinsics,
    images_folder,
    crop_box=None,
    use_mask=False,
    use_gt_depth=False,
    gt_depth_path=None,
    scale_depths=False,
    scale=1.0,
    use_gt_normal=False,
    gt_normal_path=None,
):
    """
    A modified version of `readColmapCameras()` that does image preprocessing
    if necessary. This is useful when working with dense pointclouds where we
    have one "raw" dataset that we use.

    Expected directory structure for the input:
    ```
    |- root_directory
    |   |- images_folder
    |   |   |- <image 0> # as either .jpg or .png files
    |   |   |- <image 1>
    |   |   |- ...
    |   |- masks
    |   |   |- <mask 0> # as .png files
    |   |   |- <mask 1>
    |   |   |- ...
    |   |- <gt_depth_path> # `root_directory/depths` by default
    |   |   |- <depth 0> # as .npy files
    |   |   |- <depth 1>
    |   |   |- ...
    |   |- <gt_normal_path> # `root_directory/normals` by default
    |   |   |- <normal 0> # as .npy files
    |   |   |- <normal 1>
    |   |   |- ...
    ```

    """
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))  # why do we take the transpose? -altay
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        image = Image.open(image_path)

        if crop_box is not None:
            image = image.crop(crop_box)
            image = image.resize((width, height))

        if use_mask:
            image_stem, extension = extr.name.split(".")

            # Masks directory is expected to be in the same root directory as the images folder
            mask_folder = os.path.join(os.path.dirname(images_folder), "masks")
            mask_path = os.path.join(
                mask_folder, image_stem + "_mask" + ".png"
            )  # use png masks to avoid compression
            mask = Image.open(mask_path)
            mask = np.array(mask, dtype=bool)
        else:
            mask = None

        if use_gt_depth:
            image_stem, extension = extr.name.split(".")

            if gt_depth_path == "depths":
                depth_folder = os.path.join(os.path.dirname(images_folder), "depths")
            else:
                depth_folder = gt_depth_path

            depth_path = os.path.join(
                depth_folder, image_stem + ".npy"
            )  # saving as numpy arrays
            depth = np.load(depth_path)

            # If we do not multiply the poses with the scale, we need to divide the depths by it
            if scale_depths:
                depth = depth / scale
        else:
            depth = None

        if use_gt_normal:
            image_stem, extension = extr.name.split(".")

            if gt_normal_path == "normals":
                normal_folder = os.path.join(os.path.dirname(images_folder), "normals")
            else:
                normal_folder = gt_normal_path

            normal_path = os.path.join(
                normal_folder, image_stem + ".npy"
            )  # saving as numpy arrays
            normal = np.load(normal_path)
        else:
            normal = None

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            mask=mask,
            gt_depth=depth,
            gt_normal=normal,
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


# TODO implement ground truth depth reading here for depth regularization
# TODO implement ground truth normal reading here for normal regularization
def readDenseCloudSceneInfo(
    path, images, eval, llffhold=8, use_mask=False, use_gt_depth=False
):
    """
    Custom function implementation to read SLAM extrinsics and
    dense 3D point clouds generated by repeatedly backprojecting
    depth predictions from a monocular depth prediction model.

    The implementation is based on `readColmapSceneInfo()` from the original repo
    here: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/dataset_readers.py
    """
    cameras_extrinsic_file = os.path.join(path, "poses.txt")
    cameras_intrinsic_file = os.path.join(path, "intrinsics.txt")

    cam_intrinsics, crop_box, scale = read_densecloud_intrinsics(cameras_intrinsic_file)
    cam_extrinsics = read_densecloud_extrinsics(cameras_extrinsic_file, scale)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readDenseCloudCameras(
        cam_extrinsics, cam_intrinsics, reading_dir, crop_box, use_mask
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if use_mask:
        print("Using masking to compute the loss!")

    # TODO decide whether we need this for our purposes
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(
        train_cam_infos
    )  # used to compute cameras_extent which is used in densification strategy

    # We only have the dense pointcloud as .ply
    ply_path = os.path.join(path, "cloud.ply")
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        scene_scale=scale,
    )
    return scene_info


def readDenseCloudSceneInfoColmap(
    path,
    images,
    eval,
    llffhold=8,
    use_mask=False,
    use_gt_depth=False,
    gt_depth_path=None,
    scale_depths=False,
    use_gt_normal=False,
    gt_normal_path=None,
):
    """
    Custom function to read in custom dense pointclouds and
    corresponding COLMAP poses.
    """

    cameras_intrinsic_file = os.path.join(path, "intrinsics.txt")
    cam_intrinsics, crop_box, scale = read_densecloud_intrinsics(cameras_intrinsic_file)

    try:
        cameras_extrinsic_file = os.path.join(path, "colmap_poses.txt")
        cam_extrinsics = read_densecloud_extrinsics_colmap(
            cameras_extrinsic_file, scale, scale_depths=scale_depths
        )
        print("Using colmap_poses.txt to extract the camera extrinsics!")

    except:
        cameras_extrinsic_file = os.path.join(path, "colmap_poses.bin")
        cam_extrinsics = read_densecloud_extrinsics_colmap_binary(
            cameras_extrinsic_file, scale
        )
        print("Using colmap_poses.bin to extract the camera extrinsics!")

    reading_dir = "images" if images == None else images

    if gt_depth_path is not None:
        print(f"Reading images from {gt_depth_path}...")
    else:
        print(
            "No depth path specified, looking for folder 'depths' in the parent directory of the image folder..."
        )

    if gt_normal_path is not None:
        print(f"Reading images from {gt_normal_path}...")
    else:
        print(
            "No normal path specified, looking for folder 'normals' in the parent directory of the image folder..."
        )

    cam_infos_unsorted = readDenseCloudCameras(
        cam_extrinsics,
        cam_intrinsics,
        reading_dir,
        crop_box,
        use_mask,
        use_gt_depth,
        gt_depth_path,
        scale_depths,
        scale,
        use_gt_normal,
        gt_normal_path,
    )

    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if use_mask:
        print("Using masking to compute the loss!")

    # TODO decide whether we need this for our purposes
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(
        train_cam_infos
    )  # used to compute cameras_extent which is used in densification strategy

    # We only have the dense pointcloud as .ply
    ply_path = os.path.join(path, "cloud.ply")
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        scene_scale=scale,
    )
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "DenseCloud": readDenseCloudSceneInfo,
    "DenseCloudColmap": readDenseCloudSceneInfoColmap,
}
