import numpy as np
from torch.utils.data import Dataset, Subset
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Optional, List

import os
import sys

from scipy.spatial.transform import Rotation as R




class EphemerisDataset(Dataset):
    def __init__(self, path, DBL=False, time_len=200, spherical=True, train=True):
        self.time_len_in = time_len
        self.time_len = time_len
        self.type = torch.float64 if DBL else torch.float32
        
        self.train = train

        #One trajectory, K=1
        self.npy_paths = ['nbody_2_20_15.npy',]
        
        self.process_files(path)
    
        
    def process_files(self, path):
        bodies_motion_data_list = []
        try:
            for _path in self.npy_paths:
                path = os.path.join('nbody','2body_ds', _path)
                bodies_motion_data_list.append(np.load(path))
        except:
            print('file not found')
            sys.exit(1)
        # shape is 2*4, n_steps 
        bodies_motion_data = np.stack(bodies_motion_data_list, axis=0)
        
        #use 2000 steps
        if self.train:
            bodies_motion_data = bodies_motion_data[:,:,0:2000]
        else:
            bodies_motion_data = bodies_motion_data[:,:,2000:4000]
            
        K, n4, n_step = bodies_motion_data.shape
        #print('n step ',b,  n_step)
        self.dataset_size = (n_step-2*self.time_len)
        
        #move time dim first
        bodies_motion_data = torch.tensor(bodies_motion_data, dtype=self.type)
        bodies_motion_data = bodies_motion_data.reshape(K, n4//4,4,n_step)
        bodies_motion_data = bodies_motion_data.permute(3,0,1,2)
        bodies_motion_data = bodies_motion_data.reshape(n_step,K,n4//4,4)
        
        
        self.positions = bodies_motion_data[:,:,:,0:2]
        self.velocities = bodies_motion_data[:,:,:,2:]
        
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        #idx = idx
        retx_in = self.positions[idx:idx+self.time_len_in]
        retx_out = self.positions[idx+self.time_len_in:idx+self.time_len_in+self.time_len]
        
        retv_in = self.velocities[idx:idx+self.time_len_in]
        retv_out = self.velocities[idx+self.time_len_in:idx+self.time_len_in+self.time_len]


        return retx_in, retv_in, retx_out, retv_out, idx


class EphemerisDataModule(pl.LightningDataModule):
    def __init__(self, path, bs=1,  **kwargs):
        super().__init__()
        self.bs = bs

        self.train_dataset = EphemerisDataset(path,train=True, **kwargs)
        self.val_dataset = EphemerisDataset(path,train=False, **kwargs)
        

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.bs, shuffle=True, drop_last=True, num_workers=12)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.bs, shuffle=False, drop_last=True)
        return val_loader
