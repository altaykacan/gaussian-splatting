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

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t
                    )
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.use_mask = False  # masking flag to remove pixels based off of a precomputed segmentation map to ignore certain pixels
        self.use_gt_depth = False  # flag to determine whether depth predictions are used to regularize the training
        self.use_log_loss_depth = False  # flag to determine whether the logarithm is used instead of the standard L1 loss for the depth regularization term
        self.use_tv_loss_depth = False  # flag to determine whether the total variation loss is used for the depths
        self.gt_depth_path = "depths"  # path for depths, default looks for "depths" in the parent directory of image directory
        self.scale_depths = False  # flag to determine whether poses are scaled to the depth predictions (False), or whether the depth predictions are scaled to the poses (True)
        self.use_inverse_depth = False # flag to determine whether the inverse depths are used, when inverse depths are used, there is no maximum/minimum depth thresholding # TODO decide on whether we keep
        self.use_gt_normal = False  # flag to determine whether normal predictions/estimates are used to regularize the training
        self.use_tv_loss_normal = False  # flag to determine whether the total variation loss is used for the normals
        self.gt_normal_path = "normals"  # path for normals, default looks for "normals" in the parent directory of image directory
        self.init_from_normals = False # flag to determine whether the 3D gaussians are initialized from normal values stored in the initial pointcloud

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        self.lambda_depth = 0.2  # Factor to multiply the depth regularization term
        self.lambda_tv_depth = 0.1  # Factor to multiply the total variation loss used to smoothen the depth losses
        self.lambda_normal = 0.1  # Factor to multiply the normal regularization term
        self.lambda_tv_normal = 0.1  # Factor to multiply the total variation loss used to smoothen the normal losses
        self.max_gt_depth = 50.0  # Maximum depth threshold in approximately meters (depends on the depth prediction/measurement method), depth regularization only computed depth values below this value
        self.min_gt_depth = 0.0  # Minimum depth threshold in approximately meters
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
