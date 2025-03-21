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
from lib.gaussian_renderer import render, network_gui
import sys
from lib.scene import Scene, GaussianModel
from lib.utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from lib.config import cfg
import torch.distributed as dist

from lib.train.trainers import make_trainer_single
from lib.train import make_optimizer
from lib.train import make_recorder
from lib.train import make_lr_scheduler

from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_pretrain
from lib.evaluators import make_evaluator
from lib.networks import make_network
from lib.datasets import make_data_loader


def direct_collate(x):
    return x


def training(cfg, gaussians: GaussianModel):
    scene = Scene(cfg, gaussians)
    gaussians.training_setup(cfg)

    training_generator = make_data_loader(cfg,
                                          is_train=True,
                                          scene=scene,
                                          is_distributed=cfg.distributed,
                                          max_iter=cfg.ep_iter)

    if cfg.start_checkpoint:
        (model_params, first_iter) = torch.load(cfg.start_checkpoint)
        gaussians.restore(model_params, cfg)
    if cfg.skip_eval:
        val_loader = None
    else:
        val_loader = make_data_loader(cfg, is_train=False)

    optimizer = make_optimizer(cfg, gaussians)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)
    # loading checkpoint
    begin_epoch = load_model(network,
                             optimizer,
                             scheduler,
                             recorder,
                             cfg.trained_model_dir,
                             resume=cfg.resume)
    trainer = make_trainer_single(cfg, scene, gaussians, training_generator)

    # loading pretrain network
    if begin_epoch == 0 and cfg.pretrain != '':
        load_pretrain(network, cfg.pretrain)

    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        if cfg.distributed:
            training_generator.batch_sampler.sampler.set_epoch(epoch)
        training_generator.dataset.epoch = epoch

        trainer.train(epoch, training_generator, optimizer, recorder)

        scheduler.step()
        if (epoch + 1) % cfg.save_ep == 0 and cfg.local_rank == 0:
            save_model(network,
                       optimizer,
                       scheduler,
                       recorder,
                       cfg.trained_model_dir,
                       epoch)

        if (epoch + 1) % cfg.save_latest_ep == 0 and cfg.local_rank == 0:
            save_model(network,
                       optimizer,
                       scheduler,
                       recorder,
                       cfg.trained_model_dir,
                       epoch,
                       last=True)

        if not cfg.skip_eval and (epoch + 1) % cfg.eval_ep == 0 and cfg.local_rank == 0:
            trainer.val(epoch, val_loader, evaluator, recorder)


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


if __name__ == "__main__":

    print("Optimizing " + cfg.model_path)
    print("bounds_file: " + cfg.bounds_file)
    # Initialize system state (RNG)
    safe_state(cfg.quiet)
    torch.autograd.set_detect_anomaly(cfg.detect_anomaly)

    if cfg.distributed:
        cfg.local_rank = int(os.environ['RANK']) % torch.cuda.device_count()
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl",
                                             init_method="env://")
        synchronize()

    # Start GUI server, configure and run training
    if not cfg.disable_viewer:
        network_gui.init(cfg.ip, cfg.port)

    network = GaussianModel(cfg.sh_degree)
    training(cfg, network)
    if cfg.local_rank == 0:
        print('Success!')
        print('='*80)

    # All done
    print("\nTraining complete.")
