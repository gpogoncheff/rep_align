import torch.nn as nn
from torchvision.models import alexnet, resnet50
from rep_align.models.resnet_cifar10 import *
from rep_align.models.vgg_cifar10 import *


def get_model(model_spec, n_classes):
    model = None
    if model_spec == 'alexnet':
        model = alexnet()
        model.classifier[6] = nn.Linear(model.classifier[4].out_features, n_classes, bias=True)
    elif model_spec == 'resnet50':
        model = resnet50()
        model.fc = nn.Linear(2048, n_classes, bias=True)
    elif model_spec == 'resnet18_cifar10':
        model = resnet18_cifar10()
    elif model_spec == 'resnet34_cifar10':
        model = resnet34_cifar10()
    elif model_spec == 'resnet50_cifar10':
        model = resnet50_cifar10()
    elif model_spec == 'vgg11_cifar10':
        model = vgg11_bn_cifar10()
    elif model_spec == 'vgg13_cifar10':
        model = vgg13_bn_cifar10()
    elif model_spec == 'vgg16_cifar10':
        model = vgg16_bn_cifar10()
    elif model_spec == 'vgg19_cifar10':
        model = vgg19_bn_cifar10()
    if model is None:
        raise NotImplementedError(f'Model type {model_spec} has not been implemented')
    return model