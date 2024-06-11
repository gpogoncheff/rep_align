import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from lpips import LPIPS
from tqdm import tqdm


def get_activations(model, layer_id, dataset, aggregation='mean', device='cuda'):
    model.eval()
    activations = []
    def get_activation():
        def hook(model, input, output):
            if aggregation == 'mean':
                act = output.mean(dim=(-1, -2))
            elif aggregation == 'median':
                act, _ = output.median(dim=(-1, -2))
            elif aggregation == 'min':
                act, _ = output.min(dim=(-1, -2))
            elif aggregation == 'max':
                act, _ = output.max(dim=(-1, -2))
            elif aggregation == 'center':
                y, x = output.shape[-2] // 2, output.shape[-1] // 2
                act = output[..., y, x]
            else:
                act = torch.nn.Flatten()(output)
            activations.append(act.detach())
        return hook

    hook = model.get_submodule(layer_id).register_forward_hook(get_activation())
    for inputs, _ in tqdm(dataset):
        inputs = inputs.to(device)
        if inputs.ndim < 4:
            inputs = inputs.unsqueeze(0)
        with torch.no_grad():
            _ = model(inputs)
    hook.remove()
    return torch.cat(activations, dim=0).cpu()

def get_meis(m, model, layer_id, dataset, aggregation='mean', device='cuda'):
    img_activations = get_activations(model, layer_id, dataset, aggregation, device)
    n_directions = img_activations.shape[-1]
    tmp_img, _ = next(iter(dataset))
    meis = torch.zeros((n_directions, m, *tmp_img.shape))
    mei_labels = torch.zeros((n_directions, m, 1))
    for i in range(n_directions):
        direction_activations = img_activations[:, i]
        sorted_inds = torch.argsort(direction_activations, descending=True)
        direction_meis = np.moveaxis(dataset.data[sorted_inds[:m]].copy(), -1, 1)
        meis[i] = torch.from_numpy(direction_meis)
        mei_labels[i] = torch.tensor(dataset.targets)[sorted_inds[:m]].unsqueeze(-1)
    return meis, mei_labels

def get_lpips(net, imgs, img_transform, device='cuda'):
    loss_fn_alex = LPIPS(net=net, verbose=False)
    loss_fn_alex.to(device)
    lpips_losses = []
    for i, img in enumerate(imgs):
        imgs_src = imgs[i].unsqueeze(0)
        mask = torch.ones(len(imgs))
        mask[i] = 0
        imgs_cmp = imgs[mask.bool()]
        imgs_src = img_transform(imgs_src).to(device)
        imgs_cmp = img_transform(imgs_cmp).to(device)
        lpips_losses.append(loss_fn_alex(imgs_src, imgs_cmp).detach().cpu().squeeze())
    return torch.stack(lpips_losses)

def get_pairwise_lpips(net, imgs, img_transform, batch_size=64, device='cuda'):
    loss_fn = LPIPS(net=net, verbose=False)
    loss_fn.to(device)
    n = imgs.shape[0]
    imgs_src = imgs.repeat_interleave(n, dim=0)
    imgs_cmp = imgs.repeat(n,1,1,1)
    losses = []
    for i in range(0, n, batch_size):
        a = img_transform(imgs_src[i:i+batch_size]).to(device)
        b = img_transform(imgs_cmp[i:i+batch_size]).to(device)
        losses.append(loss_fn(a, b).detach().cpu().flatten())
    return torch.concat(losses).reshape((n,n))

def l2_metric(a, b):
    # From dklindt Superposition
    sum_over_axis = tuple(np.arange(2, len(a.shape)+1))
    dist = np.sum((a[:,None] - b[None])**2, axis=sum_over_axis)
    return -np.sqrt(dist)

def get_ii_lpips(net, imgs, img_transform, batch_size=64, device='cuda'):
    mei_lpips_losses = get_pairwise_lpips(net, imgs, img_transform, batch_size, device)
    return torch.mean(mei_lpips_losses).item()

def get_ii_color(imgs, aggregate=False):
    # From dklindt Superposition
    color_channel_avgs = np.mean(imgs.numpy(), (2,3))
    sim = l2_metric(color_channel_avgs, color_channel_avgs)
    if aggregate:
        sim = np.mean(sim)
    return sim

def get_ii_label(labels, aggregate=False):
    img_labels = labels.squeeze()
    sim = (img_labels[:,None] == img_labels[None,:]).numpy()
    if aggregate:
        sim = np.mean(sim.astype(np.float32)).item()
    return sim

def evaluate_interp_index(model, layer_id, train_data=None, val_data=None, 
                          n_meis=5, mei_aggregation='center', device='cuda'):
    results = {}
    lpips_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.Lambda(lambda x: x.to(torch.float32)/255.),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    for data_id, dataset in tqdm(zip(['train', 'val'], [train_data, val_data])):
        if dataset is None:
            results[data_id] = {}
            continue
        meis, mei_labels = get_meis(n_meis, model, layer_id, dataset, 
                                    aggregation=mei_aggregation, device=device)
        ii_color, ii_lpips, ii_label = [], [], []
        for direction_i in range(len(meis)):
            ii_color.append(get_ii_color(meis[direction_i], aggregate=True))
            ii_lpips.append(get_ii_lpips('alex', meis[direction_i], lpips_transform, device=device))
            ii_label.append(get_ii_label(mei_labels[direction_i], aggregate=True))
        results[data_id] = {
            'mei_data': (meis.clone().cpu().numpy(), mei_labels.clone().cpu().numpy()),
            'ii_lpips': np.array(ii_lpips),
            'ii_color': np.array(ii_color),
            'ii_label': np.array(ii_label)
        }
    return results