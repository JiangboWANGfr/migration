import torch
from lib.utils.optimizer.radam import RAdam


_optimizer_factory = {
    'adam': torch.optim.Adam,
    'radam': RAdam,
    'sgd': torch.optim.SGD
}


def make_optimizer(cfg, net):
    params = []
    lr = cfg.train.lr
    weight_decay = cfg.train.weight_decay
    eps = cfg.train.eps

    if 'adam' in cfg.train.optim:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, weight_decay=weight_decay, eps=eps)
    else:
        optimizer = net

    return optimizer
