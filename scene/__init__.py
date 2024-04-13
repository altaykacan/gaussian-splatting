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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.gaussian_model_experimental import GaussianModelExperimental
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if args.scale_depths:
            print(
                "The flag --scale_depths is given, assuming the pointcloud has been scaled to match the scale of the poses or has the same scale. Depth predictions will be scaled if using depth regularization."
            )
        else:
            print(
                "The flag --scale_depths is not given, scaling the poses to match depth predictions and point cloud coordinates."
            )

        if os.path.exists(
            os.path.join(args.source_path, "sparse")
        ):  # loads colmap data if folder "sparse" is there
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval
            )

        elif os.path.exists(
            os.path.join(args.source_path, "transforms_train.json")
        ):  # loads Blender data if this json is there (useful for NeRF datasets)
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval
            )

        elif os.path.exists(
            os.path.join(args.source_path, "poses.txt")
        ):  # Custom callback to load dense pointclouds from orb-slam poses with EuRoC format
            print(
                "Found poses.txt, assuming custom dense point clouds are being used with EuRoC format poses!"
            )
            scene_info = sceneLoadTypeCallbacks["DenseCloud"](
                args.source_path, args.images, args.eval, use_mask=args.use_mask
            )

        elif os.path.exists(
            os.path.join(args.source_path, "colmap_poses.txt")
        ) or os.path.exists(
            os.path.join(args.source_path, "colmap_poses.bin")
        ):  # Custom callback to load dense pointclouds with colmap poses as text or binary files
            print(
                "Found colmap_poses.txt or colmap_poses.bin, assuming custom dense point clouds are being used with COLMAP format poses!"
            )
            scene_info = sceneLoadTypeCallbacks["DenseCloudColmap"](
                args.source_path,
                args.images,
                args.eval,
                use_mask=args.use_mask,
                use_gt_depth=args.use_gt_depth,
                gt_depth_path=args.gt_depth_path,
                scale_depths=args.scale_depths,
                gt_normal_path=args.gt_normal_path,
                use_gt_normal=args.use_gt_normal,
            )

        else:
            print(
                f"Couldn't recognize input file types! Please check your source path: {args.source_path}"
            )
            raise ValueError

        self.scene_scale = scene_info.scene_scale

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))  # to save cameras.json
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(
                scene_info.train_cameras
            )  # Multi-res consistent random shuffling
            random.shuffle(
                scene_info.test_cameras
            )  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )  # calls loadCam
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                )
            )
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, args.init_from_normals)

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
