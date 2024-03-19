from typing import List

from arguments import ModelParams
from . import Scene, GaussianModel
from .dataset_readers import sceneLoadTypeCallbacks

class CompositeScene(Scene):

    scenes: List[Scene]

    def __init__(self, args : ModelParams, gaussians : GaussianModel, resolution_scales=[1.0], grid_size: tuple =[4, 4]):
        super().__init__(args, gaussians, load_iteration=None, shuffle=True, resolution_scales=[1.0], scene_info=None, root_scene=True)
        self.grid_size = grid_size

