import torch
from tqdm import tqdm
from lib.utils.loss_utils import l1_loss, ssim
from lib.gaussian_renderer import render_coarse, network_gui
import sys
from lib.scene import Scene


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
    pass

def make_trainer_post(cfg, scene, gaussians, training_generator=None):
    pass