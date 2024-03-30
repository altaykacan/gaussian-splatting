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
from scene.dataset_readers import read_data
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], scene_info=None, root_scene: bool = False, scene_id: int = 0):
        """b
        Args:
            scene_info: Information about the scene, contains the pointcloud and
                lists of `CameraInfo`'s
            root_scene: Boolean flag whether this Scene instance is the first
                scene (root scene) in a CompositeScene structure
            scene_id: Integer id of the scene within a CompositeScene structure
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.root_scene = root_scene
        self.scene_id = scene_id

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {} # they are dictionaries to hold the different views for different resolution scales
        self.test_cameras = {} # different from the counterparts in scene_info

        # Only read-in data if no initial scene_info is present
        if scene_info is None:
            self.scene_info = read_data(args)

        # Modified to save according to the scene id
        if not self.loaded_iter:
            with open(self.scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, f"input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if self.scene_info.test_cameras:
                camlist.extend(self.scene_info.test_cameras)
            if self.scene_info.train_cameras:
                camlist.extend(self.scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam)) # to save cameras.json, put into dataset directory to be able to use remote viewer
            with open(os.path.join(self.model_path, f"cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # When training we pick a random viewpoint anyways, this shuffling isn't very important
        if shuffle:
            random.shuffle(self.scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(self.scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = self.scene_info.nerf_normalization["radius"]

        # Root scene does not load the train cameras
        if not root_scene:
            for resolution_scale in resolution_scales:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.train_cameras, resolution_scale, args) # calls loadCam
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.test_cameras, resolution_scale, args)

        # If this Scene is the root scene, we don't want to load any gaussians
        if not root_scene:
            if self.loaded_iter:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"))
            else:
                self.gaussians.create_from_pcd(self.scene_info.point_cloud, self.cameras_extent)

        # No need to keep scene_info in memory if this is not the root scene
        if not root_scene:
            del self.scene_info

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, f"point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]


