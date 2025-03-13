#
# Copyright (C) 2023 - 2024, Inria
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
from lib.utils.loss_utils import l1_loss, ssim
from lib.gaussian_renderer import render_coarse, network_gui
import sys
from lib.scene import Scene, GaussianModel
from lib.utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from torch.utils.data import DataLoader
from lib.config import cfg


from lib.datasets import make_data_loader


def training(dataset, opt, pipe, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    first_iter = 0
    gaussians = GaussianModel(1)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    training_generator = make_data_loader(cfg,
                                          is_train=True,
                                          scene=scene,
                                          is_distributed=cfg.distributed,
                                          max_iter=cfg.ep_iter)

    if cfg.skip_eval:
        val_loader = None
    else:
        val_loader = make_data_loader(cfg, is_train=False)

    # training_generator = DataLoader(scene.getTrainCameras(
    # ), num_workers=8, prefetch_factor=1, persistent_workers=True, collate_fn=direct_collate)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations),
                        desc="Training progress")
    first_iter += 1

    indices = None

    iteration = first_iter

    # for param_group in gaussians.optimizer.param_groups:
    #     if param_group["name"] == "xyz":
    #         param_group['lr'] = 0.0
    #     print(param_group['name'], param_group['lr'])
    while iteration < opt.iterations + 1:
        for viewpoint_batch in training_generator:
            for viewpoint_cam in viewpoint_batch:
                background = torch.rand(
                    (3), dtype=torch.float32, device="cuda")

                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()

                if network_gui.conn == None:
                    network_gui.try_connect()
                    
                iter_start.record()

                # Every 1000 its we increase the levels of SH up to a maximum degree
                if iteration % 1000 == 0:
                    gaussians.oneupSHdegree()

                # Render
                if (iteration - 1) == debug_from:
                    pipe.debug = True

                render_pkg = render_coarse(
                    viewpoint_cam, gaussians, pipe, background, indices=indices)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                    "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                # Loss
                gt_image = viewpoint_cam.original_image.cuda().float()
                if viewpoint_cam.alpha_mask is not None:
                    alpha_mask = viewpoint_cam.alpha_mask.cuda().float()
                    Ll1 = l1_loss(image * alpha_mask, gt_image)
                    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * \
                        (1.0 - ssim(image * alpha_mask, gt_image))
                else:
                    Ll1 = l1_loss(image, gt_image)
                    loss = (1.0 - opt.lambda_dssim) * Ll1 + \
                        opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                loss.backward()

                iter_end.record()

                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii)

                with torch.no_grad():
                    # Progress bar
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                    if iteration % 10 == 0:
                        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Size": f"{gaussians._xyz.size(0)}",
                                                 "Peak memory": f"{torch.cuda.max_memory_allocated(device='cuda')}"})
                        progress_bar.update(10)

                    # Log and save
                    if (iteration in saving_iterations):
                        print("\n[ITER {}] Saving Gaussians".format(iteration))
                        scene.save(iteration)

                    if iteration == opt.iterations:
                        progress_bar.close()
                        training_generator._get_iterator()._shutdown_workers()
                        return

                    # Optimizer step

                    if iteration < opt.iterations:
                        gaussians._scaling.grad[:gaussians.skybox_points, :] = 0
                        relevant = (gaussians._opacity.grad != 0).nonzero()
                        gaussians.optimizer.step(relevant)
                        gaussians.optimizer.zero_grad(set_to_none=True)

                    # if (iteration in checkpoint_iterations):
                    #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    #     torch.save((gaussians.capture(), iteration),
                    #                scene.model_path + "/chkpnt" + str(iteration) + ".pth")

                    with torch.no_grad():
                        vals, _ = gaussians.get_scaling.max(dim=1)
                        violators = vals > scene.cameras_extent * 0.1
                        violators[:gaussians.skybox_points] = False
                        gaussians._scaling[violators] = gaussians.scaling_inverse_activation(
                            gaussians.get_scaling[violators] * 0.8)

                    iteration += 1


if __name__ == "__main__":
    print("Training coarse model started")
    print("Optimizing " + cfg.model_path)
    print(cfg.model_path)

    # Initialize system state (RNG)
    safe_state(cfg.quiet)
    cfg.scaffold_file = ""
    # Start GUI server, configure and run training
    network_gui.init(cfg.ip, cfg.port)
    torch.autograd.set_detect_anomaly(cfg.detect_anomaly)
    training(cfg, cfg, cfg, cfg.save_iterations,
             cfg.checkpoint_iterations, cfg.start_checkpoint, cfg.debug_from)

    # All done
    print("\nTraining complete.")
