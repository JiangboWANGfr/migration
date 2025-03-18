import torch
from tqdm import tqdm
from lib.utils.loss_utils import l1_loss, ssim
from lib.gaussian_renderer import render_coarse, network_gui
from lib.utils.general_utils import safe_state, get_expon_lr_func
from lib.gaussian_renderer import render, network_gui
from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights
from lib.gaussian_renderer import render_post
import sys
from lib.scene import Scene
import math


def make_trainer(cfg, scene, gaussians, training_generator=None):

    bg_color = [1, 1, 1] if cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    first_iter = 0
    progress_bar = tqdm(range(first_iter, cfg.iterations),
                        desc="Training progress")
    first_iter += 1
    indices = None
    iteration = first_iter

    while iteration < cfg.iterations + 1:
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
                if (iteration - 1) == cfg.debug_from:
                    cfg.debug = True

                render_pkg = render_coarse(
                    viewpoint_cam, gaussians, cfg, background, indices=indices)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                    "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                # Loss
                gt_image = viewpoint_cam.original_image.cuda().float()
                if viewpoint_cam.alpha_mask is not None:
                    alpha_mask = viewpoint_cam.alpha_mask.cuda().float()
                    Ll1 = l1_loss(image * alpha_mask, gt_image)
                    loss = (1.0 - cfg.lambda_dssim) * Ll1 + cfg.lambda_dssim * \
                        (1.0 - ssim(image * alpha_mask, gt_image))
                else:
                    Ll1 = l1_loss(image, gt_image)
                    loss = (1.0 - cfg.lambda_dssim) * Ll1 + \
                        cfg.lambda_dssim * (1.0 - ssim(image, gt_image))
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
                    if (iteration in cfg.save_iterations):
                        print("\n[ITER {}] Saving Gaussians".format(iteration))
                        scene.save(iteration)

                    if iteration == cfg.iterations:
                        progress_bar.close()
                        training_generator._get_iterator()._shutdown_workers()
                        return

                    # Optimizer step

                    if iteration < cfg.iterations:
                        gaussians._scaling.grad[:gaussians.skybox_points, :] = 0
                        relevant = (gaussians._opacity.grad != 0).nonzero()
                        gaussians.optimizer.step(relevant)
                        gaussians.optimizer.zero_grad(set_to_none=True)

                    with torch.no_grad():
                        vals, _ = gaussians.get_scaling.max(dim=1)
                        violators = vals > scene.cameras_extent * 0.1
                        violators[:gaussians.skybox_points] = False
                        gaussians._scaling[violators] = gaussians.scaling_inverse_activation(
                            gaussians.get_scaling[violators] * 0.8)

                    iteration += 1


def make_trainer_single(cfg, scene, gaussians, training_generator=None):
    first_iter = 0
    bg_color = [1, 1, 1] if cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    depth_l1_weight = get_expon_lr_func(
        cfg.depth_l1_weight_init, cfg.depth_l1_weight_final, max_steps=cfg.iterations)

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    progress_bar = tqdm(range(first_iter, cfg.iterations),
                        desc="Training progress")
    first_iter += 1

    indices = None

    iteration = first_iter

    while iteration < cfg.iterations + 1:
        for viewpoint_batch in training_generator:
            for viewpoint_cam in viewpoint_batch:
                background = torch.rand(
                    (3), dtype=torch.float32, device="cuda")

                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()

                if not cfg.disable_viewer:
                    if network_gui.conn == None:
                        network_gui.try_connect()
                    while network_gui.conn != None:
                        try:
                            net_image_bytes = None
                            custom_cam, do_training, cfg.convert_SHs_python, cfg.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                            if custom_cam != None:
                                if keep_alive:
                                    net_image = render(
                                        custom_cam, gaussians, cfg, background, scaling_modifer, indices=indices)["render"]
                                else:
                                    net_image = render(custom_cam, gaussians, cfg, background, scaling_modifer, indices=indices)[
                                        "depth"].repeat(3, 1, 1)
                                net_image_bytes = memoryview((torch.clamp(
                                    net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                            network_gui.send(net_image_bytes, cfg.source_path)
                            if do_training and ((iteration < int(cfg.iterations)) or not keep_alive):
                                break
                        except Exception as e:
                            network_gui.conn = None

                iter_start.record()

                gaussians.update_learning_rate(iteration)

                # Every 1000 its we increase the levels of SH up to a maximum degree
                if iteration % 1000 == 0:
                    gaussians.oneupSHdegree()

                # Render
                if (iteration - 1) == cfg.debug_from:
                    cfg.debug = True
                render_pkg = render(viewpoint_cam, gaussians, cfg,
                                    background, indices=indices, use_trained_exp=True)
                image, invDepth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                    "depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                # Loss
                gt_image = viewpoint_cam.original_image.cuda()
                if viewpoint_cam.alpha_mask is not None:
                    alpha_mask = viewpoint_cam.alpha_mask.cuda()
                    image *= alpha_mask

                Ll1 = l1_loss(image, gt_image)
                Lssim = (1.0 - ssim(image, gt_image))
                photo_loss = (1.0 - cfg.lambda_dssim) * \
                    Ll1 + cfg.lambda_dssim * Lssim
                loss = photo_loss.clone()
                Ll1depth_pure = 0.0
                if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                    mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                    depth_mask = viewpoint_cam.depth_mask.cuda()

                    Ll1depth_pure = torch.abs(
                        (invDepth - mono_invdepth) * depth_mask).mean()
                    Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
                    loss += Ll1depth
                    Ll1depth = Ll1depth.item()
                else:
                    Ll1depth = 0

                loss.backward()
                iter_end.record()

                with torch.no_grad():
                    # Progress bar
                    ema_loss_for_log = 0.4 * photo_loss.item() + 0.6 * ema_loss_for_log
                    ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
                    if iteration % 10 == 0:
                        progress_bar.set_postfix(
                            {"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}", "Size": f"{gaussians._xyz.size(0)}"})
                        progress_bar.update(10)

                    # Log and save
                    if (iteration in cfg.checkpoint_iterations):
                        print("\n[ITER {}] Saving Gaussians".format(iteration))
                        scene.save(iteration)
                        print("peak memory: ",
                              torch.cuda.max_memory_allocated(device='cuda'))

                    if iteration == cfg.iterations:
                        progress_bar.close()
                        return

                    # Densification
                    if iteration < cfg.densify_until_iter:
                        # Keep track of max radii in image-space for pruning
                        gaussians.max_radii2D[visibility_filter] = torch.max(
                            gaussians.max_radii2D[visibility_filter], radii)
                        gaussians.add_densification_stats(
                            viewspace_point_tensor, visibility_filter)

                        if iteration > cfg.densify_from_iter and iteration % cfg.densification_interval == 0:
                            gaussians.densify_and_prune(
                                cfg.densify_grad_threshold, 0.005, scene.cameras_extent)

                        if iteration % cfg.opacity_reset_interval == 0 or (cfg.white_background and iteration == cfg.densify_from_iter):
                            print("-----------------RESET OPACITY!-------------")
                            gaussians.reset_opacity()

                    # Optimizer step
                    if iteration < cfg.iterations:
                        gaussians.exposure_optimizer.step()
                        gaussians.exposure_optimizer.zero_grad(
                            set_to_none=True)

                        if gaussians._xyz.grad != None and gaussians.skybox_locked:
                            gaussians._xyz.grad[:gaussians.skybox_points, :] = 0
                            gaussians._rotation.grad[:gaussians.skybox_points, :] = 0
                            gaussians._features_dc.grad[:gaussians.skybox_points, :, :] = 0
                            gaussians._features_rest.grad[:gaussians.skybox_points, :, :] = 0
                            gaussians._opacity.grad[:gaussians.skybox_points, :] = 0
                            gaussians._scaling.grad[:gaussians.skybox_points, :] = 0

                        if gaussians._opacity.grad != None:
                            relevant = (
                                gaussians._opacity.grad.flatten() != 0).nonzero()
                            relevant = relevant.flatten().long()
                            if (relevant.size(0) > 0):
                                gaussians.optimizer.step(relevant)
                            else:
                                gaussians.optimizer.step(relevant)
                                print("No grads!")
                            gaussians.optimizer.zero_grad(set_to_none=True)

                    if not cfg.skip_scale_big_gauss:
                        with torch.no_grad():
                            vals, _ = gaussians.get_scaling.max(dim=1)
                            violators = vals > scene.cameras_extent * 0.02
                            if gaussians.scaffold_points is not None:
                                violators[:gaussians.scaffold_points] = False
                            gaussians._scaling[violators] = gaussians.scaling_inverse_activation(
                                gaussians.get_scaling[violators] * 0.8)

                    if (iteration in cfg.checkpoint_iterations):
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                        torch.save((gaussians.capture(), iteration),
                                   scene.model_path + "/chkpnt" + str(iteration) + ".pth")

                    iteration += 1



def make_trainer_post(cfg, scene, gaussians, training_generator=None):
    first_iter = 0

    if cfg.start_checkpoint:
        (model_params, first_iter) = torch.load(cfg.start_checkpoint)
        gaussians.restore(model_params, cfg)

    bg_color = [1, 1, 1] if cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, cfg.iterations),
                        desc="Training progress")
    first_iter += 1

    indices = None

    iteration = first_iter

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
                limit = math.pow(
                    2, sample * (math.log2(limmax) - math.log2(limmin)) + math.log2(limmin))
                scale = 1

                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()

                # Then with blending training
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
                    parent_indices=parent_indices,
                    interpolation_weights=interpolation_weights,
                    num_node_kids=num_siblings,
                    use_trained_exp=True,
                )
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                    "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                # Loss
                gt_image = viewpoint_cam.original_image.cuda()
                if viewpoint_cam.alpha_mask is not None:
                    Ll1 = l1_loss(
                        image * viewpoint_cam.alpha_mask.cuda(), gt_image)
                    loss = (1.0 - cfg.lambda_dssim) * Ll1 + cfg.lambda_dssim * \
                        (1.0 - ssim(image * viewpoint_cam.alpha_mask.cuda(), gt_image))
                else:
                    Ll1 = l1_loss(image, gt_image)
                    loss = (1.0 - cfg.lambda_dssim) * Ll1 + \
                        cfg.lambda_dssim * (1.0 - ssim(image, gt_image))

                loss.backward()

                iter_end.record()

                with torch.no_grad():
                    # Progress bar
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                    if iteration % 10 == 0:
                        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Size": f"{gaussians._xyz.size(0)}",
                                                 "Peak memory": f"{torch.cuda.max_memory_allocated(device='cuda')}"})
                        progress_bar.update(10)

                    # Log and save
                    if (iteration in cfg.save_iterations):
                        print("\n[ITER {}] Saving Gaussians".format(iteration))
                        scene.save(iteration)
                        print("peak memory: ",
                              torch.cuda.max_memory_allocated(device='cuda'))

                    if iteration == cfg.iterations:

                        progress_bar.close()
                        return

                    # Optimizer step
                    if iteration < cfg.iterations:

                        if gaussians._xyz.grad != None:
                            if gaussians.skybox_points != 0 and gaussians.skybox_locked:  # No post-opt for skybox
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
                        gaussians.optimizer.zero_grad(set_to_none=True)

                    if (iteration in cfg.checkpoint_iterations):
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                        torch.save((gaussians.capture(), iteration),
                                   scene.model_path + "/chkpnt" + str(iteration) + ".pth")

                    iteration += 1
