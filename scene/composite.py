from typing import List

import torch
import numpy as np

from arguments import ModelParams
from scene import Scene, GaussianModel
from scene.dataset_readers import read_data

class CompositeScene(Scene):

    scenes: List[Scene]
    combined_gaussians: GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, resolution_scales = [1.0], grid_size = 4):
        # scene_info is None so it is read from the source path
        super().__init__(args, gaussians, load_iteration=None, shuffle=True, resolution_scales=resolution_scales, scene_info=None, root_scene=True)
        self.grid_size = grid_size

        self._extract_poses()

        # No need to keep scene_info once the root scene has been divided
        del self.scene_info

    def _extract_poses(self):
        """
        Extracts the pose information contained as a list of
        `CameraInfo`'s in `self.scene_info` and saves them to dictionaries
        that contain numpy arrays for easier processing.
        """
        # List of rotations (3x3) and translation (3) vectors
        self.train_poses = {"R": [], "T": []}
        self.test_poses = {"R": [], "T": []}

        for train_cam_info in self.scene_info.train_cameras:
            self.train_poses["R"].append(train_cam_info.R)
            self.train_poses["T"].append(train_cam_info.T)

        for test_cam_info in self.scene_info.test_cameras:
            self.test_poses["R"].append(test_cam_info.R)
            self.test_poses["T"].append(test_cam_info.T)

        # Rotations are saved as (num_frames, 3, 3) arrays, translations as (num_frames, 3)
        for key, value in self.train_poses.items():
            self.train_poses[key] = np.array(value)

        for key, value in self.test_poses.items():
            self.test_poses[key] = np.array(value)


    def _compute_boundaries(self):
        pass

    def _divide_scene(self):
        pass

    def increment_gaussians(self, current_gaussians):
        # No need to keep gradient information when we are combining the gaussians
        with torch.no_grad:
            pass


class IncrementalScene(Scene):
    pass
