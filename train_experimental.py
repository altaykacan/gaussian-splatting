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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l1_loss_mask, ssim_mask, total_variation_loss, log_depth_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModelExperimental
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    # dataset, opt, and pipe are all GroupParams objects, which are empty classes with key-value pairs for the respective settings
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModelExperimental(dataset.sh_degree)
    scene = Scene(dataset, gaussians) # changed -altay
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # Scale the max and min ground truth depth thresholds if we scale depths and pointcloud to match poses
    if dataset.scale_depths:
        opt.max_gt_depth = opt.max_gt_depth / scene.scene_scale
        opt.min_gt_depth = opt.min_gt_depth / scene.scene_scale

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if dataset.use_gt_depth:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, return_depth=True) # TODO maybe can add masking here (cuda) as well -altay
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg) # TODO maybe can add masking here (cuda) as well -altay

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        if dataset.use_mask:
            mask = viewpoint_cam.mask.cuda()

            Ll1 = l1_loss_mask(image, gt_image, mask)
            ssim_loss = 1.0 - ssim_mask(image, gt_image, mask)
        else:
            mask = None

            Ll1 = l1_loss(image, gt_image)
            ssim_loss = 1.0 - ssim(image, gt_image)

        if dataset.use_gt_depth:
            gt_depth = viewpoint_cam.gt_depth.cuda()
            depth = render_pkg["render_depth"]

            mask_depth = torch.logical_and(gt_depth < opt.max_gt_depth, gt_depth > opt.min_gt_depth)

            if mask is not None:
                mask_depth = torch.logical_and(mask_depth, mask)

            if dataset.use_log_loss_depth:
                depth_loss = log_depth_loss(depth, gt_depth, mask_depth)
            else:
                depth_loss = l1_loss_mask(depth, gt_depth, mask_depth)
        else:
            gt_depth = None
            depth_loss = torch.Tensor([0.0]).cuda()

        if dataset.use_tv_loss and dataset.use_gt_depth:
            tv_loss = total_variation_loss(depth, mask_depth)
        else:
            tv_loss = torch.tensor([0.0]).cuda()


        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + opt.lambda_depth * depth_loss + opt.lambda_tv * tv_loss
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                depth_loss=depth_loss,
                tv_loss=tv_loss,
                opt=opt,
                )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold) # Ne

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", "debug_" + unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, depth_loss=None, tv_loss=None, opt=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

        tb_writer.add_scalar('train_loss_patches/depth_loss', depth_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/tv_loss', tv_loss.item(), iteration)


    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 40, 5)]})

        if iteration == 40000:
            print("hol'up")

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    # Adding custom tensorboard visualizations to show rendered depth and normal maps
                    render_results = renderFunc(viewpoint, scene.gaussians, *renderArgs, return_depth=True, return_normal=True)

                    image_vis = torch.clamp(render_results["render"], 0.0, 1.0)
                    depth_vis = render_results["render_depth"]
                    inv_depth_vis = 1 / (depth_vis + 0.00001)
                    normals = render_results["render_normal"]

                    gt_image_vis = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    inv_depth_norm = (inv_depth_vis - inv_depth_vis.min()) / (inv_depth_vis.max() - inv_depth_vis.min())

                    # Normalize the rgb values channel-wise
                    normals_norm = torch.zeros_like(normals)
                    for channel in range(3):
                        channel_min = normals[channel, :, :].min()
                        channel_max = normals[channel, :, :].max()
                        normalized_channel = (normals[channel, :, :] - channel_min) / (channel_max - channel_min)
                        normals_norm[channel, :, :] = normalized_channel

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image_vis[None], global_step=iteration)

                        tb_writer.add_images(config['name'] + "_view_{}_depths/inv_depth".format(viewpoint.image_name), inv_depth_norm[None, None], global_step=iteration) # [None, None] prepends an empty dimension for batch and channel
                        tb_writer.add_images(config['name'] + "_view_{}_normals/render".format(viewpoint.image_name), normals_norm[None], global_step=iteration)

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image_vis[None], global_step=iteration)

                            if viewpoint.gt_depth is not None:
                                gt_depth_vis = viewpoint.gt_depth.cuda()
                                gt_inv_depth = 1 / (gt_depth_vis+ 0.00001)
                                gt_inv_depth_norm = (gt_inv_depth - gt_inv_depth.min()) / (gt_inv_depth.max() - gt_inv_depth.min())
                                tb_writer.add_images(config['name'] + "_view_{}_depths/ground_truth".format(viewpoint.image_name), gt_inv_depth_norm[None, None], global_step=iteration)

                                if viewpoint.mask is not None:
                                    mask_depth = torch.logical_and(gt_depth_vis < opt.max_gt_depth, gt_depth_vis > opt.min_gt_depth)
                                    mask_depth = torch.logical_and(mask_depth, viewpoint.mask)
                                else:
                                    mask_depth = torch.logical_and(gt_depth_vis < opt.max_gt_depth, gt_depth_vis > opt.min_gt_depth)

                                tb_writer.add_images(config['name'] + "_view_{}_masks/depth".format(viewpoint.image_name), mask_depth[None, None], global_step=iteration)

                                if viewpoint.mask is not None:
                                    mask_moveable = viewpoint.mask.cuda()
                                    tb_writer.add_images(config['name'] + "_view_{}_masks/moveable".format(viewpoint.image_name), mask_moveable[None, None], global_step=iteration)

                    l1_test += l1_loss(image_vis, gt_image_vis).mean().double()
                    psnr_test += psnr(image_vis, gt_image_vis).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":

    # The answer
    # torch.manual_seed(42)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet) # overrides sys.stdout for nice printing and sets deterministic state

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # lp.extract() returns a GroupParams object which is a dummy class to hold dictionaries of config settings
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")