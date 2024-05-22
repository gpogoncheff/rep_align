import os
import numpy as np
import pandas as pd
import skimage.io
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class CadenaSessionDataset(Dataset):
    N_NEURONS = 255
    def __init__(self, session_id, response_dir, img_dir, img_transform=None, response_transform=None):
        self.response_metadata = pd.read_csv(os.path.join(response_dir, f'{session_id}.csv'))
        self.neuron_masks = np.memmap(os.path.join(response_dir, f'{session_id}_masks.dat'), mode='r',
                                      dtype=np.float32, shape=(len(self.response_metadata), self.N_NEURONS))
        self.neuron_responses = np.memmap(os.path.join(response_dir, f'{session_id}.dat'), mode='r',
                                          dtype=np.float32, shape=(len(self.response_metadata), self.N_NEURONS))
        self.img_dir = img_dir
        self.img_transform = img_transform
        self.response_transform = response_transform

    def __len__(self):
        return len(self.response_metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_fpath = os.path.join(self.img_dir, '{:06}.png'.format(self.response_metadata.iloc[idx].image_id))
        img = skimage.io.imread(img_fpath)
        img = torch.from_numpy(img)
        neuron_mask = np.array(self.neuron_masks[idx])
        neuron_mask = torch.from_numpy(neuron_mask)
        spike_response = np.array(self.neuron_responses[idx])
        spike_response = torch.from_numpy(spike_response)

        if self.response_transform:
            spike_response = self.response_transform(spike_response)
        if self.img_transform:
            img = self.img_transform(img)

        return img, spike_response, neuron_mask


class CadenaV4Data:
    N_NEURONS = 255
    def __init__(self, data_fpath, batch_size, shuffle=True):
        self.data = h5py.File(data_fpath)
        self.datasets = session_datasets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def __init__(self, session_datasets, batch_size, num_workers, shuffle=True):
        self.datasets = session_datasets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.dataloaders = [
            DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle) for dataset in self.datasets
        ]
        self.reset_iterators()
        self.length = int(np.ceil(np.sum([len(ds) for ds in self.datasets])/batch_size))
        self.index = 0

    def __len__(self):
        return self.length

    def reset_iterators(self):
        self.dataloader_iterators = [iter(dataloader) for dataloader in self.dataloaders]
        self.dataloader_indices = np.arange(len(self.dataloader_iterators))
        if self.shuffle:
            self.dataloader_indices = np.random.permutation(self.dataloader_indices)

    def get_next(self):
        no_yield = True
        while no_yield:
            if len(self.dataloader_indices) == 0:
                break
            try:
                img, spike_responses, neuron_mask = next(self.dataloader_iterators[self.dataloader_indices[0]])
                self.dataloader_indices = np.roll(self.dataloader_indices, -1)
                no_yield = False
            except StopIteration:
                self.dataloader_indices = self.dataloader_indices[1:]
        if no_yield:
            #self.reset_iterators()
            return None, None, None
        return img, spike_responses, neuron_mask
