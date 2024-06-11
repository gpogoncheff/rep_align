import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class NSDVoxels(Dataset):
    def __init__(self, data_root, voxel_rois, transforms=None):
        self.data_root = data_root

        if type(voxel_rois) == str:
            voxel_rois = [voxel_rois]
        self.voxel_rois = voxel_rois
        self.voxel_data = []
        for voxel_roi in self.voxel_rois:
            self.voxel_data.append(np.load(os.path.join(data_root, 'brain_activities', f'{voxel_roi}.npy')))
        self.voxel_data = np.hstack(self.voxel_data)
        self.voxel_data = torch.from_numpy(self.voxel_data).to(torch.float32)
        self.img_paths = np.load(os.path.join(data_root, 'preprocessed', 'img_paths.npy'))
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)
        return img, self.voxel_data[idx]