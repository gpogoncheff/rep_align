import math
import numpy as np
import torch.nn as nn

# Learning rate schedulers derived from 
# https://github.com/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb
# https://github.com/huyvnphan/PyTorch_CIFAR10/blob/master/cifar10_models/resnet.py
class LearningRateScheduler:
    def __init__(self):
        pass
    def step(self):
        pass
    def get_lr(self):
        raise NotImplementedError

class ConstantLR(LearningRateScheduler):
    def __init__(self, lr):
        super(ConstantLR, self).__init__()
        self.lr = lr

    def get_lr(self):
        return self.lr
    
class LinearWarmupCosineDecayLR(LearningRateScheduler):
    def __init__(self, warmup_start_lr, base_lr, warmup_steps, max_steps, eta_min):
        super(LinearWarmupCosineDecayLR, self).__init__()
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eta_min = eta_min
        self.current_step = 0

    def step(self):
        self.current_step += 1

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.warmup_start_lr + \
                (self.current_step * ((self.base_lr - self.warmup_start_lr) / (self.warmup_steps - 1)))
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * \
            (1 + math.cos(math.pi * (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)))

class LinearDecayLR(LearningRateScheduler):
    def __init__(self, lr_init, max_steps):
        super(LinearDecayLR, self).__init__()
        self.lr_init = lr_init
        self.max_steps = max_steps

    def step(self):
        self.current_step += 1

    def get_lr(self):
        return self.lr_init * (1 - (self.current_step / self.max_steps))


def step_lr(steps, scale=0.1):
    def get_step_lr_scale(step):
        return scale**(step//steps)
    return get_step_lr_scale

def cosine_decay_lr(steps):
    def get_cosine_decay_lr_scale(step):
        return np.cos(0.5 * np.pi * step / (steps - 1))
    return get_cosine_decay_lr_scale