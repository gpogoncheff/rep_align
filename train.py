import os
import sys
import time
import yaml
import argparse
import math
import numpy as np
import madry
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50, alexnet
from rep_align.models.model_factory import get_model
from rep_align.utils.lr_scheduling import *
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt


def evaluate(model, eval_data, epoch, device='cuda'):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    n, total_loss, total_correct = 0, 0, 0
    with torch.no_grad():
        with tqdm(eval_data, unit='batch') as eval_progress:
            eval_progress.set_description(f'eval  {epoch}')
            for img, target in eval_progress:
                img = img.to(device)
                target = target.to(device)
                output = model(img)
                n += len(target)
                total_loss += loss_fn(output, target).item()
                total_correct += torch.sum(torch.argmax(output, dim=-1) == target).item()
                avg_loss = (total_loss/n)
                acc = (total_correct/n)
                eval_progress.set_postfix(loss='{:.4f}'.format(avg_loss), acc='{:.4f}'.format(acc))
    if WANDB:
        wandb.log({'val_loss': avg_loss, 'val_acc': acc, 'epoch': epoch})
    return avg_loss, acc

def train_epoch(model, train_data, loss_fn, optimizer, lr_scheduler, epoch, device='cuda'):
    model.train()
    n, running_loss, running_correct = 0, 0, 0
    with tqdm(train_data, unit='batch') as epoch_progress:
        epoch_progress.set_description(f'train {epoch}')
        for img, target in epoch_progress:
            img = img.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            new_lr = lr_scheduler.get_lr()
            for group in optimizer.param_groups:
                group['lr'] = new_lr
                if WANDB:
                    wandb.log({'lr': new_lr, 'lr_step': lr_scheduler.current_step})
            n += len(target)
            running_loss += loss.item()*len(target)
            running_correct += torch.sum(torch.argmax(output, dim=-1) == target).item()
            epoch_progress.set_postfix(loss='{:.4f}'.format((running_loss/n)), acc='{:.4f}'.format((running_correct/n)))
    avg_loss = (running_loss/n)
    acc = (running_correct/n)
    if WANDB:
        wandb.log({'train_loss': avg_loss, 'train_acc': acc, 'epoch': epoch})
    return avg_loss, acc

def train(model, train_data, evaluation_data, epochs, loss_fn, optimizer, lr_scheduler,
          save_dir, save_freq=float('inf'), save_best=True, device='cuda'):
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_loss = float('inf')
    base_val_acc = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_data, loss_fn, optimizer, lr_scheduler, epoch, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_loss, val_acc = evaluate(model, evaluation_data, epoch, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        model_ckpt_data = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'best_val_loss': best_val_loss,
            'wandb_run_name': None if not WANDB else wandb.run.name,
        }
        if ((epoch+1) % save_freq) == 0:
            torch.save(model_ckpt_data, os.path.join(save_dir, f'epoch_{epoch}.pt'))
        if save_best and val_loss < best_val_loss:
            torch.save(model_ckpt_data, os.path.join(save_dir, 'best_loss.pt'))
            best_val_loss = val_loss
        if save_best and val_acc > base_val_acc:
            torch.save(model_ckpt_data, os.path.join(save_dir, 'best_acc.pt'))
            base_val_acc = val_acc
      
    return (train_losses, train_accs), (val_losses, val_accs)

WANDB = False
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    config_path = args.config_path
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    device = config_data['system']['device']
    num_workers = config_data['system']['num_workers']
    epochs = config_data['optimization']['epochs']
    batch_size = config_data['optimization']['batch_size']
    model_spec = config_data['model']
    dataset = config_data['dataset']
    optimizer_spec = config_data['optimization']['optimizer']
    optimizer_params = config_data['optimization']['optimizer_params']
    lr_scheduling_spec = config_data['optimization']['lr_scheduling']
    lr_sheduling_params = config_data['optimization']['lr_scheduling_params']
    save_dir = config_data['logging']['save_dir']
    assert not os.path.exists(save_dir), 'Save directory already exists'
    os.mkdir(save_dir)
    if config_data['logging']['wandb']:
        WANDB = True
        wandb.init(
            project='repalign',
            config=dict(yaml=config_data)
        )

    train_transforms = []
    train_transform_parms = config_data['train_transforms']
    if train_transform_parms['resize'] != -1:
        train_transforms.append(transforms.Resize(train_transform_parms['resize']))
    if len(train_transform_parms['random_crop'].keys()) > 0:
        crop_params = train_transform_parms['random_crop']
        train_transforms.append(transforms.RandomCrop(
                                    crop_params['size'],
                                    padding=crop_params['padding']))
    if train_transform_parms['horizontal_flip']:
         train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(transforms.Normalize(
        train_transform_parms['norm_mean'],
        train_transform_parms['norm_std'],
    ))

    val_transforms = []
    val_transform_parms = config_data['val_transforms']
    if val_transform_parms['resize'] != -1:
        val_transforms.append(transforms.Resize(val_transform_parms['resize']))
    if len(val_transform_parms['random_crop'].keys()) > 0:
        crop_params = val_transform_parms['random_crop']
        val_transforms.append(transforms.RandomCrop(
                                    crop_params['size'],
                                    padding=crop_params['padding']))
    if val_transform_parms['horizontal_flip']:
         val_transforms.append(transforms.RandomHorizontalFlip())
    val_transforms.append(transforms.ToTensor())
    val_transforms.append(transforms.Normalize(
        val_transform_parms['norm_mean'],
        val_transform_parms['norm_std'],
    ))

    if dataset == 'cifar10':
        n_classes = 10
        train_transforms = transforms.Compose(train_transforms)
        val_transforms = transforms.Compose(val_transforms)
        train_dataset = CIFAR10(root='./data', train=True, transform=train_transforms, download=True)
        eval_dataset = CIFAR10(root='./data', train=False, transform=val_transforms, download=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if model_spec == 'resnet50_madry':
        model = madry.ResNet50()
    else:
        model = get_model(model_spec, n_classes)
    model.to(device)

    if lr_scheduling_spec == 'constant':
        lr_scheduler = ConstantLR(lr_sheduling_params['lr_init'])
    elif lr_scheduling_spec == 'linear_warmup_cosine_decay':
        lr_scheduler = LinearWarmupCosineDecayLR(lr_sheduling_params['warmup_start_lr'], 
                                                 lr_sheduling_params['base_lr'], 
                                                 lr_sheduling_params['warmup_steps']*len(train_dataloader), 
                                                 lr_sheduling_params['max_steps']*len(train_dataloader), 
                                                 lr_sheduling_params['eta_min'])

    if optimizer_spec == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
    elif optimizer_spec == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)

    loss_fn = nn.CrossEntropyLoss()

    train_hist, val_hist = train(model=model, train_data=train_dataloader, evaluation_data=eval_dataloader, 
                                 epochs=epochs, loss_fn=loss_fn, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                 save_dir=save_dir, device=device)

    if WANDB:
        wandb.finish()