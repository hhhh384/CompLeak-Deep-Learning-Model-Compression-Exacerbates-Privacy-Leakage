import numpy as np
import torch
import random
from torch.nn import init
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import _LRScheduler

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_model(model_type, num_cls, input_dim):
    if model_type == "resnet18":
        from net.resnet import resnet18
        model = resnet18(pretrained=False, num_classes=num_cls)

    elif model_type == 'resnet50':
        from net.resnet import resnet50
        model = resnet50()

    elif model_type == 'vgg16':
        from  net.vgg import vgg16_bn
        model = vgg16_bn(pretrained=False, num_classes=num_cls)
    
    elif model_type == "mobilenetv2":
        from net.mobilenetv2 import MobileNetV2
        model = MobileNetV2()

    elif model_type == "transformer":
        from net.transformer import Transformer
        model = Transformer(input_dim=num_cls, output_dim=2)

    else:
        print(model_type)
        raise ValueError
    return model


def get_optimizer(optimizer_name, parameters, lr, weight_decay=5e-4):
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    elif optimizer_name == "":
        optimizer = None
    else:
        print(optimizer_name)
        raise ValueError
    return optimizer


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
