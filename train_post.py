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
from lib.gaussian_renderer import render_post
import sys
from lib.scene import Scene, GaussianModel
from lib.utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
import math
from lib.config import cfg
import torch.distributed as dist


from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights

def direct_collate(x):
    return x

def training(cfg, gaussians: GaussianModel):
    first_iter = 0
    gaussians.active_sh_degree = cfg.sh_degree
    scene = Scene(cfg, gaussians, resolution_scales = [1], create_from_hier=True)
    gaussians.training_setup(cfg, our_adam=False)
    if cfg.start_checkpoint:
        (model_params, first_iter) = torch.load(cfg.start_checkpoint)
        gaussians.restore(model_params, cfg)

    bg_color = [1, 1, 1] if cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, cfg.iterations), desc="Training progress")
    first_iter += 1

    indices = None

    iteration = first_iter
    training_generator = DataLoader(scene.getTrainCameras(), num_workers = 8, prefetch_factor = 1, persistent_workers = True, collate_fn=direct_collate)

    limit = 0.001

    render_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
    parent_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
    nodes_for_render_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
    interpolation_weights = torch.zeros(gaussians._xyz.size(0)).float().cuda()
    num_siblings = torch.zeros(gaussians._xyz.size(0)).int().cuda()
    to_render = 0

    limmax = 0.1
    limmin = 0.005

    while iteration < cfg.iterations + 1:
        for viewpoint_batch in training_generator:
            for viewpoint_cam in viewpoint_batch:

                sample = torch.rand(1).item()
                limit = math.pow(2, sample * (math.log2(limmax) - math.log2(limmin)) + math.log2(limmin))
                scale = 1

                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()

                #Then with blending training
                iter_start.record()

                gaussians.update_learning_rate(iteration)

                # Every 1000 its we increase the levels of SH up to a maximum degree
                if iteration % 1000 == 0:
                    gaussians.oneupSHdegree()

                to_render = expand_to_size(
                    gaussians.nodes,
                    gaussians.boxes,
                    limit * scale,
                    viewpoint_cam.camera_center,
                    torch.zeros((3)),
                    render_indices,
                    parent_indices,
                    nodes_for_render_indices)
                
                indices = render_indices[:to_render].int()
                node_indices = nodes_for_render_indices[:to_render]

                get_interpolation_weights(
                    node_indices,
                    limit * scale,
                    gaussians.nodes,
                    gaussians.boxes,
                    viewpoint_cam.camera_center.cpu(),
                    torch.zeros((3)),
                    interpolation_weights,
                    num_siblings
                )

                # Render
                if (iteration - 1) == cfg.debug_from:
                    cfg.debug = True

                render_pkg = render_post(
                    viewpoint_cam, 
                    gaussians, 
                    cfg, 
                    background, 
                    render_indices=indices,
                    parent_indices = parent_indices,
                    interpolation_weights = interpolation_weights,
                    num_node_kids = num_siblings,
                    use_trained_exp=True,
                    )
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


                # Loss
                gt_image = viewpoint_cam.original_image.cuda()
                if viewpoint_cam.alpha_mask is not None:
                    Ll1 = l1_loss(image * viewpoint_cam.alpha_mask.cuda(), gt_image)
                    loss = (1.0 - cfg.lambda_dssim) * Ll1 + cfg.lambda_dssim * (1.0 - ssim(image * viewpoint_cam.alpha_mask.cuda(), gt_image))
                else:
                    Ll1 = l1_loss(image, gt_image) 
                    loss = (1.0 - cfg.lambda_dssim) * Ll1 + cfg.lambda_dssim * (1.0 - ssim(image, gt_image))

                loss.backward()

                iter_end.record()

                with torch.no_grad():
                    # Progress bar
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                    if iteration % 10 == 0:
                        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Size": f"{gaussians._xyz.size(0)}", "Peak memory": f"{torch.cuda.max_memory_allocated(device='cuda')}"})
                        progress_bar.update(10)

                    # Log and save
                    if (iteration in cfg.save_iterations):
                        print("\n[ITER {}] Saving Gaussians".format(iteration))
                        scene.save(iteration)
                        print("peak memory: ", torch.cuda.max_memory_allocated(device='cuda'))

                    if iteration == cfg.iterations:
                            
                        progress_bar.close()
                        return

                    # Optimizer step
                    if iteration < cfg.iterations:

                        if gaussians._xyz.grad != None:
                            if gaussians.skybox_points != 0 and gaussians.skybox_locked: #No post-opt for skybox
                                gaussians._xyz.grad[-gaussians.skybox_points:, :] = 0
                                gaussians._rotation.grad[-gaussians.skybox_points:, :] = 0
                                gaussians._features_dc.grad[-gaussians.skybox_points:, :, :] = 0
                                gaussians._features_rest.grad[-gaussians.skybox_points:, :, :] = 0
                                gaussians._opacity.grad[-gaussians.skybox_points:, :] = 0
                                gaussians._scaling.grad[-gaussians.skybox_points:, :] = 0
                            
                            gaussians._xyz.grad[gaussians.anchors, :] = 0
                            gaussians._rotation.grad[gaussians.anchors, :] = 0
                            gaussians._features_dc.grad[gaussians.anchors, :, :] = 0
                            gaussians._features_rest.grad[gaussians.anchors, :, :] = 0
                            gaussians._opacity.grad[gaussians.anchors, :] = 0
                            gaussians._scaling.grad[gaussians.anchors, :] = 0
                        

                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none = True)

                    if (iteration in cfg.checkpoint_iterations):
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                        torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

                    iteration += 1


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(cfg.model_path))
    os.makedirs(cfg.model_path, exist_ok=True)
    with open(os.path.join(cfg.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

if __name__ == "__main__":
    
    print("Optimizing " + cfg.model_path)
    # Initialize system state (RNG)
    safe_state(cfg.quiet)
    
    if cfg.distributed:
        cfg.local_rank = int(os.environ['RANK']) % torch.cuda.device_count()
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl",
                                             init_method="env://")
        synchronize()


    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(cfg.detect_anomaly)
    gaussians = GaussianModel(cfg.sh_degree)
    training(cfg, gaussians)

    print("\nTraining complete.")
