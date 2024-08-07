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
from utils.loss_utils import (
    l1_loss,
    ssim,
    l1_loss_mask,
    ssim_mask,
    total_variation_loss,
    log_depth_loss,
    constant_opacity_loss,
    disk_loss,
    dna_loss,
)
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, psnr_mask
from utils.camera_utils import perturb_viewpoint
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
):
    ##########
    #  Setup
    ##########
    # dataset, opt, and pipe are all GroupParams objects, which are empty classes with key-value pairs for the respective settings
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # Scale the max and min ground truth depth thresholds if we scale depths and pointcloud to match poses
    if dataset.scale_depths:
        opt.max_gt_depth = opt.max_gt_depth / scene.scene_scale
        opt.min_gt_depth = opt.min_gt_depth / scene.scene_scale

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    prev_dna_loss = None

    ##########
    # Training loop
    ##########
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
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

        # Pick a random Camera for every iteration
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        ##########
        # Render
        ##########
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # TODO maybe can add masking here (to the cuda function) as well
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            return_depth=dataset.use_gt_depth,
            return_normal=dataset.use_gt_normal,
            return_is_road=dataset.use_gt_road_mask,
        )

        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        ##########
        # Loss
        ##########
        gt_image = viewpoint_cam.original_image.cuda()

        # Photometric loss
        if dataset.use_mask:
            mask = viewpoint_cam.mask.cuda()
            Ll1 = l1_loss_mask(image, gt_image, mask)
            ssim_loss = 1.0 - ssim_mask(image, gt_image, mask)
        else:
            mask = None
            Ll1 = l1_loss(image, gt_image)
            ssim_loss = 1.0 - ssim(image, gt_image)

        # Depth regularization
        if dataset.use_gt_depth:
            gt_depth = viewpoint_cam.gt_depth.cuda()
            depth = render_pkg["render_depth"]

            if dataset.use_inverse_depth:
                # Inverse depth does not require masking
                mask_depth = torch.ones_like(depth)
                depth = 1 / (depth + 0.000001)
                gt_depth = 1 / (gt_depth + 0.000001)
            else:
                mask_depth = torch.logical_and(gt_depth < opt.max_gt_depth, gt_depth > opt.min_gt_depth)

            # Moveable object mask, pixels on non-movable objects are True
            if mask is not None:
                mask_depth = torch.logical_and(mask_depth, mask)

            # Alternative to L1 loss is to use the log(1 + l1_loss(depth, gt_depth))
            if dataset.use_log_loss_depth:
                depth_loss = log_depth_loss(depth, gt_depth, mask_depth)
            else:
                depth_loss = l1_loss_mask(depth, gt_depth, mask_depth)

            # Total variation loss to enforce smoothness
            if dataset.use_tv_loss_depth:
                tv_loss_depth = total_variation_loss(depth, mask_depth)
            else:
                tv_loss_depth = torch.tensor([0.0]).cuda()
        else:
            mask_depth = None
            depth_loss = torch.Tensor([0.0]).cuda()
            tv_loss_depth = torch.tensor([0.0]).cuda()

        # Normal regularization
        if dataset.use_gt_normal and not dataset.use_dna:
            gt_normal = viewpoint_cam.gt_normal.float().cuda()
            normal = render_pkg["render_normal"]

            # Moveable object mask, pixels on non-movable objects are True
            if mask is not None:
                mask_normal = mask

            normal_loss = l1_loss_mask(normal, gt_normal, mask_normal)
        else:
            normal_loss = torch.Tensor([0.0]).cuda()
            tv_loss_normal = torch.Tensor([0.0]).cuda()

        # Direct normal alignment
        if dataset.use_gt_normal and dataset.use_dna and (iteration > opt.apply_dna_from_iter - 1) and (iteration < opt.apply_dna_until_iter + 1):
            # Mask for the collection of Gaussians, not the renderings
            dna_mask = visibility_filter & gaussians.get_is_road.squeeze()
            # dna_mask = gaussians.get_is_road.squeeze()
            dna_loss_term = dna_loss(gaussians, dna_mask)
        else:
            dna_loss_term = torch.Tensor([0.0]).cuda()

        # Total variation loss to enforce smoothness
        if dataset.use_gt_normal and dataset.use_tv_loss_normal:
            normal = render_pkg["render_normal"]
            if mask is not None:
                mask_normal = mask
            tv_loss_normal = total_variation_loss(normal, mask_normal)
        else:
            tv_loss_normal = torch.Tensor([0.0]).cuda()

        # # Alpha entropy regularization
        # entropy = render_pkg["entropy"]
        # if dataset.use_entropy_regularization and (iteration > opt.apply_entropy_losses_from_iter) and (iteration < opt.apply_entropy_losses_until_iter):
        #     gt_entropy = torch.zeros_like(entropy).cuda()
        # else:
        #     gt_entropy = entropy.clone().cuda()

        # entropy_loss = l1_loss(entropy, gt_entropy)

        # Constant opacity term
        if dataset.use_constant_opacity_loss:
            opacity_mask = visibility_filter & gaussians.get_is_road.squeeze()
            opacities = gaussians.get_opacity[opacity_mask]

            opacity_loss = constant_opacity_loss(opacities, opt.opacity_target)
        else:
            opacity_loss = torch.Tensor([0.0]).cuda()

        # Disk loss
        if dataset.use_disk_loss:
            scales = gaussians.get_scaling[visibility_filter & gaussians.get_is_road.squeeze()]
            disk_loss_term = disk_loss(scales)
        else:
            disk_loss_term = torch.Tensor([0.0]).cuda()

        # Road loss
        if dataset.use_gt_road_mask:
            gt_road_mask = viewpoint_cam.gt_road_mask.cuda().float()
            road_mask = render_pkg["render_is_road"].float()

            # Last 'mask' makes the loss ignore pixels on movable objects
            road_loss = l1_loss_mask(road_mask, gt_road_mask, mask)
        else:
            road_loss = torch.Tensor([0.0]).cuda()

        loss = (
            (1.0 - opt.lambda_dssim) * Ll1
            + opt.lambda_dssim * ssim_loss
            + opt.lambda_depth * (depth_loss + opt.lambda_tv_depth * tv_loss_depth)
            + opt.lambda_normal * (normal_loss + opt.lambda_tv_normal * tv_loss_normal)
            + opt.lambda_normal * (dna_loss_term + opt.lambda_tv_normal * tv_loss_normal)
            + opt.lambda_opacity * opacity_loss
            # + opt.lambda_entropy * entropy_loss
            + opt.lambda_disk * disk_loss_term
            + opt.lambda_road_mask * road_loss
        )

        loss.backward()

        # Zero out certain gradients
        if opt.dna_zero_grad and dataset.use_gt_normal and (iteration > opt.apply_dna_from_iter - 1) and (iteration < opt.apply_dna_until_iter + 1):
            ignore_mask = gaussians.get_is_road.squeeze()

            # No rotation and position gradients for road Gaussians
            # gaussians._rotation.grad[ignore_mask] = 0.0
            gaussians._xyz.grad[ignore_mask] = 0.0

            if dataset.use_dna:
                if prev_dna_loss is None:
                    prev_dna_loss = dna_loss_term

                if dna_loss_term > (prev_dna_loss + 0.001):
                    print(f"Previous dna loss ({prev_dna_loss:0.8f}) is not the same as the current dna loss ({dna_loss_term:0.8f}) even though you specified to zero the gradients")
                    prev_dna_loss = dna_loss_term

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration % 500 == 1:
                print("=========")
                print("image_id: ", viewpoint_cam.uid)
                print("L1 loss: ", Ll1)
                print("SSIM loss: ", ssim_loss)
                print("Depth loss: ", depth_loss)
                print("Normal loss: ", normal_loss)
                print("Opacity loss: ", opacity_loss)
                print("DNA loss: ", dna_loss_term)
                print("Road loss: ", road_loss)
                # print("Entropy loss: ", entropy_loss)
                print("Radii max: ", radii.max())
                print("Gaussian scales max: ", gaussians.get_scaling.max())
                print("Number of gaussians: ", gaussians.get_scaling.shape[0])

            ##########
            # Log and save
            ##########
            l1_loss_for_val = l1_loss if not dataset.use_mask else l1_loss_mask
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss_for_val,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                depth_loss=depth_loss,
                tv_loss_depth=tv_loss_depth,
                normal_loss=normal_loss,
                tv_loss_normal=tv_loss_normal,
                opacity_loss=opacity_loss,
                # entropy_loss=entropy_loss,
                disk_loss=disk_loss_term,
                dna_loss=dna_loss_term,
                road_loss=road_loss,
                opt=opt,
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                print("\n[ITER {}] Number of Gaussians: {}".format(iteration, gaussians.get_xyz.shape[0]))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        dataset.minimum_opacity,
                        scene.cameras_extent,
                        size_threshold,
                        dont_prune_road=dataset.dont_prune_road,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

        # Normal resetting
        if dataset.reset_normals and iteration % opt.reset_normals_interval == 0:
            reset_normal_mask = gaussians.get_is_road.squeeze()
            gaussians.reset_normals(reset_normal_mask)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", "debug_" + unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    depth_loss=None,
    tv_loss_depth=None,
    normal_loss=None,
    tv_loss_normal=None,
    opacity_loss=None,
    # entropy_loss=None,
    disk_loss=None,
    dna_loss = None,
    road_loss= None,
    opt=None,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

        if depth_loss is not None:
            tb_writer.add_scalar("train_loss_patches/depth_loss", depth_loss.item(), iteration)

        if tv_loss_depth is not None:
            tb_writer.add_scalar("train_loss_patches/tv_loss_depth", tv_loss_depth.item(), iteration)

        if normal_loss is not None:
            tb_writer.add_scalar("train_loss_patches/normal_loss", normal_loss.item(), iteration)

        if tv_loss_normal is not None:
            tb_writer.add_scalar("train_loss_patches/tv_loss_normal", tv_loss_normal.item(), iteration)

        if opacity_loss is not None:
            tb_writer.add_scalar("train_loss_patches/constant_opacity_loss", opacity_loss.item(), iteration)

        # if entropy_loss is not None:
            # tb_writer.add_scalar("train_loss_patches/entropy_loss", entropy_loss.item(), iteration)

        if disk_loss is not None:
            tb_writer.add_scalar("train_loss_patches/disk_loss", disk_loss.item(), iteration)

        if dna_loss is not None:
            tb_writer.add_scalar("train_loss_patches/dna_loss", dna_loss.item(), iteration)

        if road_loss is not None:
            tb_writer.add_scalar("train_loss_patches/road_loss", road_loss.item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 40, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    # Adding custom tensorboard visualizations to show rendered depth and normal maps
                    render_results = renderFunc(
                        viewpoint,
                        scene.gaussians,
                        *renderArgs,
                        return_depth=True,
                        return_normal=True,
                        return_gt_normal = True,
                        return_is_road = True,
                    )

                    image = torch.clamp(render_results["render"], 0.0, 1.0)
                    depth = render_results["render_depth"]
                    inv_depth = 1 / (depth + 0.000001)
                    normal = render_results["render_normal"]
                    gt_normal_render = render_results.get("render_gt_normal", None)
                    is_road_render = render_results.get("render_is_road", None)
                    # entropy = render_results["entropy"]

                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    inv_depth_norm = (inv_depth - inv_depth.min()) / (
                        inv_depth.max() - inv_depth.min()
                    )

                    # Convert [-1,1] range of the normals to [0,1] for float value visualization
                    normal_norm = (normal + 1) / 2
                    if gt_normal_render is not None:
                        gt_normal_render_norm = (gt_normal_render + 1) / 2

                    # entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min())

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )

                        # tb_writer.add_images(
                        #     config["name"]
                        #     + "_view_{}_entropy/entropy".format(viewpoint.image_name),
                        #     entropy[None],
                        #     global_step=iteration,
                        # )

                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}_depths/inv_depth".format(viewpoint.image_name),
                            inv_depth_norm[None, None],
                            global_step=iteration,
                        )  # [None, None] prepends an empty dimension for batch and channel

                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}_normals/render".format(viewpoint.image_name),
                            normal_norm[None],
                            global_step=iteration,
                        )

                        if gt_normal_render is not None:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}_normals/gt_render".format(viewpoint.image_name),
                                gt_normal_render_norm[None],
                                global_step=iteration,
                            )

                        if is_road_render is not None:
                            is_road_render = is_road_render[0, :, :] # take first channel only
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}_is_road/render".format(viewpoint.image_name),
                                is_road_render[None, None],
                                global_step=iteration,
                            )

                        # Renderings for perturbed viewpoints
                        # perturbed_viewpoints = perturb_viewpoint(viewpoint, scene.cameras_extent)
                        # for perturbed_name, perturbed_viewpoint in perturbed_viewpoints.items():
                        #     pt_render_results = renderFunc(
                        #         perturbed_viewpoint,
                        #         scene.gaussians,
                        #         *renderArgs,
                        #         return_depth=True,
                        #         return_normal=True,
                        #     )

                        #     pt_image = torch.clamp(pt_render_results["render"], 0.0, 1.0)
                        #     pt_depth = pt_render_results["render_depth"]
                        #     pt_inv_depth = 1 / (pt_depth + 0.000001)
                        #     pt_normal = pt_render_results["render_normal"]
                        #     # pt_entropy = pt_render_results["entropy"]

                        #     pt_inv_depth_norm = (pt_inv_depth - pt_inv_depth.min()) / (
                        #         pt_inv_depth.max() - pt_inv_depth.min()
                        #     )

                        #     # Convert [-1,1] range of the normals to [0,1] for float value visualization
                        #     pt_normal_norm = (pt_normal + 1) / 2

                        #     # pt_entropy = (pt_entropy - pt_entropy.min()) / (pt_entropy.max() - pt_entropy.min())

                        #     tb_writer.add_images(
                        #         config["name"]
                        #         + "_view_{}_perturbed/render/{}".format(viewpoint.image_name, perturbed_name),
                        #         pt_image[None],
                        #         global_step=iteration,
                        #     )

                        #     # tb_writer.add_images(
                        #     #     config["name"]
                        #     #     + "_view_{}_perturbed/entropy/{}".format(viewpoint.image_name, perturbed_name),
                        #     #     pt_entropy[None],
                        #     #     global_step=iteration,
                        #     # )

                        #     tb_writer.add_images(
                        #         config["name"]
                        #         + "_view_{}_perturbed/depths/{}".format(viewpoint.image_name, perturbed_name),
                        #         pt_inv_depth_norm[None, None],
                        #         global_step=iteration,
                        #     )  # [None, None] prepends an empty dimension for batch and channel
                        #     tb_writer.add_images(
                        #         config["name"]
                        #         + "_view_{}_perturbed/normals/{}".format(viewpoint.image_name, perturbed_name),
                        #         pt_normal_norm[None],
                        #         global_step=iteration,
                        #     )

                        # Save ground truth values and used masks only for the first iteration
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )

                            if viewpoint.mask is not None:
                                mask_moveable = viewpoint.mask.cuda()
                                tb_writer.add_images(
                                    config["name"]
                                    + "_view_{}_masks/moveable".format(
                                        viewpoint.image_name
                                    ),
                                    mask_moveable[None, None],
                                    global_step=iteration,
                                )

                            if viewpoint.gt_depth is not None:
                                gt_depth = viewpoint.gt_depth.cuda()
                                gt_inv_depth = 1 / (gt_depth + 0.000001)
                                gt_inv_depth_norm = (
                                    gt_inv_depth - gt_inv_depth.min()
                                ) / (gt_inv_depth.max() - gt_inv_depth.min())
                                tb_writer.add_images(
                                    config["name"]
                                    + "_view_{}_depths/ground_truth".format(
                                        viewpoint.image_name
                                    ),
                                    gt_inv_depth_norm[None, None],
                                    global_step=iteration,
                                )

                                if viewpoint.mask is not None:
                                    mask_depth = torch.logical_and(
                                        gt_depth < opt.max_gt_depth,
                                        gt_depth > opt.min_gt_depth,
                                    )
                                    mask_depth = torch.logical_and(
                                        mask_depth, viewpoint.mask
                                    )
                                else:
                                    mask_depth = torch.logical_and(
                                        gt_depth < opt.max_gt_depth,
                                        gt_depth > opt.min_gt_depth,
                                    )

                                tb_writer.add_images(
                                    config["name"]
                                    + "_view_{}_masks/depth".format(
                                        viewpoint.image_name
                                    ),
                                    mask_depth[None, None],
                                    global_step=iteration,
                                )

                            if viewpoint.gt_normal is not None:
                                gt_normal = viewpoint.gt_normal.cuda()
                                gt_normal_norm = (
                                    gt_normal + 1
                                ) / 2  # map from [-1,1] to [0,1]
                                tb_writer.add_images(
                                    config["name"]
                                    + "_view_{}_normals/ground_truth".format(
                                        viewpoint.image_name
                                    ),
                                    gt_normal_norm[None],
                                    global_step=iteration,
                                )

                    if viewpoint.mask is not None:
                        mask = viewpoint.mask.cuda()
                        l1_test += l1_loss(image, gt_image, mask).mean().double()
                        psnr_test += psnr_mask(image, gt_image, mask).mean().double()
                        ssim_test += ssim_mask(image, gt_image, mask).double()
                    else:
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                        ssim_test += ssim(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                ssim_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {}".format(
                        iteration, config["name"], l1_test, psnr_test, ssim_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - ssim", ssim_test, iteration
                    )
            torch.cuda.empty_cache()

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar(
                "total_points", scene.gaussians.get_xyz.shape[0], iteration
            )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # The answer
    torch.manual_seed(42)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10, 1_000, 5_000, 7_000, 10_000, 15_000, 20_000, 25_000, 30_000, 40_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)  # overrides sys.stdout for nice printing and sets deterministic state

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # lp.extract() returns a GroupParams object which is a dummy class to hold dictionaries of config settings
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )

    # All done
    print("\nTraining complete.")
