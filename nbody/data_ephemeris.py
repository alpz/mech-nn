import numpy as np
from torch.utils.data import Dataset, Subset
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Optional, List

import re
import os
from io import StringIO

from scipy.spatial.transform import Rotation as R
import ipdb

#from IPython import embed


class EphemerisDataset(Dataset):
    def __init__(self, path='datasets', DBL=True, time_len=200, num_bodies=10, spherical=True, train=True):
        #TODO load files
        self.time_len_in = time_len
        self.time_len = time_len
        self.type = torch.float64 if DBL else torch.float32
        self.num_bodies = num_bodies
        
        self.train = train
        
        self.process_files(path)
        
    def process_files(self, path='datasets'):
        x_path = os.path.join(path, 'ephemerides', 'astro_pos_720_sun.npy')
        v_path = os.path.join(path, 'ephemerides', 'astro_vel_720_sun.npy')

        X = np.load(x_path)
        V = np.load(v_path)
        
        print(X.shape)
        print(V.shape)

        X = torch.tensor(X, dtype=self.type)
        V = torch.tensor(V, dtype=self.type)
        #move time dim first
        X = X.permute(1,0,2)
        V = V.permute(1,0,2)

        data_len = X.shape[0]
        train_len = int(0.7*data_len)
        val_len = int(0.3*data_len)
        if self.train:
            X = X[:train_len]
            V = V[:train_len]
        else:
            X = X[train_len:train_len+val_len]
            V = V[train_len:train_len+val_len]

        n_step = X.shape[0]#//self.subsample
        self.dataset_size = (n_step-2*self.time_len+1)
        
        self.positions = X
        self.velocities = V
        

        self.positions = self.positions[:,:self.num_bodies,:]#.unsqueeze(1)
        self.velocities = self.velocities[:,:self.num_bodies,:]#.unsqueeze(1)

        

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        idx = idx
        retx_in = self.positions[idx:idx+self.time_len_in]
        retv_in = self.velocities[idx:idx+self.time_len_in]

        retx_out = self.positions[idx+self.time_len_in:idx+self.time_len_in+self.time_len]
        retv_out = self.velocities[idx+self.time_len_in:idx+self.time_len_in+self.time_len]

        return retx_in, retv_in, retx_out, retv_out, idx


class EphemerisDataModule(pl.LightningDataModule):
    def __init__(self, path, bs=1, **kwargs):
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
