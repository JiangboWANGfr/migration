import sys
from argparse import ArgumentParser, Namespace
from .yacs import CfgNode as CN
import argparse
import os
import numpy as np
from . import yacs
cfg = CN()

os.environ['workspace'] = './'

cfg.workspace = os.environ['workspace']

cfg.save_result = False
cfg.clear_result = False
cfg.save_tag = 'default'
# module
# cfg.train_dataset_module = 'lib.datasets.dtu.neus'
# cfg.test_dataset_module = 'lib.datasets.dtu.neus'
# cfg.val_dataset_module = 'lib.datasets.dtu.neus'
# cfg.network_module = 'lib.neworks.neus.neus'
# cfg.loss_module = 'lib.train.losses.neus'
# cfg.evaluator_module = 'lib.evaluators.neus'

# experiment name
cfg.exp_name = 'gitbranch_hello'
cfg.exp_name_tag = ''
cfg.pretrain = ''

cfg.local_rank = 0

# network
cfg.distributed = False

# task
cfg.task = 'hello'

# gpus
cfg.gpus = list(range(1))
# if load the pretrained network
cfg.resume = True

# epoch
cfg.ep_iter = -1
cfg.save_ep = 1
cfg.save_latest_ep = 1000
cfg.eval_ep = 1
cfg.log_interval = 1

cfg.task_arg = CN()
# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()
cfg.train.epoch = 0
cfg.train.num_workers = 8
cfg.train.collator = 'default'
cfg.train.batch_sampler = 'default'
cfg.train.sampler_meta = CN({})
cfg.train.shuffle = True
cfg.train.eps = 1e-8

# use adam as default
cfg.train.optim = 'Adam'
cfg.train.lr = 5e-4
cfg.train.weight_decay = 0.
cfg.train.scheduler = CN({'type': 'multi_step1', 'milestones': [
                         80, 120, 200, 240], 'gamma': 0.5})
cfg.train.batch_size = 4

# test
cfg.test = CN()
cfg.test.batch_size = 1
cfg.test.collator = 'default'
cfg.test.epoch = -1
cfg.test.num_workers = 0
cfg.test.batch_sampler = 'default'
cfg.test.sampler_meta = CN({})

# trained model
cfg.trained_model_dir = os.path.join(os.environ['workspace'], 'trained_model')
cfg.clean_tag = 'debug'

# recorder
cfg.record_dir = os.path.join(os.environ['workspace'], 'record')

# result
cfg.result_dir = os.path.join(os.environ['workspace'], 'result')

# evaluation
cfg.skip_eval = True
cfg.fix_random = False


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
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument(
                        "--" + key, default=value, action="store_true")
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
        self._exp_name = ""
        self._images = "images"
        self._alpha_masks = ""
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        # Include the left half of the test images in the train set to optimize exposures
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        self.skip_scale_big_gauss = False
        self.hierarchy = ""
        self.pretrained = ""
        self.skybox_num = 0
        self.scaffold_file = ""
        self.bounds_file = ""
        self.skybox_locked = False
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
        self.position_lr_init = 0.00002
        self.position_lr_final = 0.0000002
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.001
        self.exposure_lr_final = 0.0001
        self.exposure_lr_delay_steps = 5000
        self.exposure_lr_delay_mult = 0.001
        self.percent_dense = 0.0001
        self.lambda_dssim = 0.2
        self.densification_interval = 300
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.015
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
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


def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')
    # assign the gpus
    if -1 not in cfg.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(
            [str(gpu) for gpu in cfg.gpus])



def make_cfg(args):
    def merge_cfg(cfg_file, cfg):
        with open(cfg_file, 'r') as f:
            current_cfg = yacs.load_cfg(f)

        if 'parent_cfg' in current_cfg.keys():
            cfg = merge_cfg(current_cfg.parent_cfg, cfg)
            cfg.merge_from_other_cfg(current_cfg)
        else:
            cfg.merge_from_other_cfg(current_cfg)
        return cfg
    cfg_ = merge_cfg(args.cfg_file, cfg)
    try:
        index = args.opts.index('other_opts')
        cfg_.merge_from_list(args.opts[:index])
    except:
        cfg_.merge_from_list(args.opts)
        
    parse_cfg(cfg_, args)
    return cfg_


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
# parser.add_argument('--model_path', type=str, default="")
parser.add_argument('--bounds_file', type=str,
                    default="./data/small_city/camera_calibration/chunks/0_0/")
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()

if len(args.type) > 0:
    cfg.task = "run"
cfg = make_cfg(args)
if args.bounds_file:
    cfg['bounds_file'] = args.bounds_file
