import numpy as np
from torch.utils.data import Dataset, Subset
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Optional, List
import pandas as pd

import astropy.coordinates
import re
import os
from io import StringIO

from scipy.spatial.transform import Rotation as R
import ipdb

from sklearn.preprocessing import StandardScaler
from IPython import embed


def read_skip_header_footer(path):
    start = 0
    lines = []
    with open(path) as f:
        for line in f:
            if '$$EOE' in line.upper():
                start = 0
            elif start:
                lines.append(line)
            elif '$$SOE' in line.upper():
                start = 1
    string = ''.join(lines)
    return string


def cartesian_to_spherical(x, y, z, v_x, v_y, v_z):
    """
    Utility function (jitted) to convert cartesian to spherical.
    This function should eventually result in Coordinate Transformation Graph!
    """
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    theta = np.arctan2(hxy, z)
    phi = np.arctan2(y, x)
    n1 = x**2 + y**2
    n2 = n1 + z**2
    v_r = (x * v_x + y * v_y + z * v_z) / np.sqrt(n2)
    v_th = (z * (x * v_x + y * v_y) - n1 * v_z) / (n2 * np.sqrt(n1))
    v_p = -1 * (v_x * y - x * v_y) / n1

    return r, theta, phi, v_r, v_th, v_p 

def spherical_to_cartesian(r, th, p, v_r, v_th, v_p):
    """
    Utility function (jitted) to convert spherical to cartesian.
    This function should eventually result in Coordinate Transformation Graph!
    """
    x = r * np.cos(p) * np.sin(th)
    y = r * np.sin(p) * np.sin(th)
    z = r * np.cos(th)
    v_x = (
        np.sin(th) * np.cos(p) * v_r
        - r * np.sin(th) * np.sin(p) * v_p
        + r * np.cos(th) * np.cos(p) * v_th
    )
    v_y = (
        np.sin(th) * np.sin(p) * v_r
        + r * np.cos(th) * np.sin(p) * v_th
        + r * np.sin(th) * np.cos(p) * v_p
    )
    v_z = np.cos(th) * v_r - r * np.sin(th) * v_th

    return x, y, z, v_x, v_y, v_z


class EphemerisDataset(Dataset):
    def __init__(self, path, DBL=False, time_len=200, spherical=True, train=True):
        #TODO load files
        self.time_len_in = time_len
        self.time_len = time_len
        self.type = torch.float64 if DBL else torch.float32
        
        self.train = train
        #if train:
        #    #self.npy_paths = ['nbody_2_10_10.npy','nbody_2_10_20.npy','nbody_2_10_30.npy','nbody_2_20_20.npy','nbody_2_10_5.npy',
        #    #         'nbody_2_20_50.npy','nbody_2_20_40.npy']
        #    
        #    self.npy_paths = ['nbody_2_20_15.npy',]
        #    
        #    #self.npy_paths = ['nbody_2_10_10.npy','nbody_2_12_12.npy','nbody_2_15_15.npy','nbody_2_5_5.npy','nbody_2_22_22.npy',
        #    #         'nbody_2_25_25.npy','nbody_2_30_30.npy']
        #else:
        #    self.npy_paths = ['nbody_2_20_15.npy',]
        self.npy_paths = ['nbody_2_20_15.npy',]
        #self.npy_paths = ['nbody_4_10.npy',]
        #self.npy_paths = ['nbody_10_8.npy',]
        #self.npy_paths = ['nbody_8_5.npy',]
        
        self.process_files(path)
        
    def convert_to_sph(self, data):
        print(data.shape)
        T, B, D = data.shape
        r,t,p, vr,vt,vp = cartesian_to_spherical(data[:,:,0], data[:,:,1], data[:,:,2],
                                                data[:,:,3], data[:,:,4], data[:,:,5])
        data = np.stack([r,t, p, vr,vt,vp], axis=2)
        
        print('out', data.shape)
        return data
    
        
    def process_files(self, path):
        bodies_motion_data_list = []
        try:
            for _path in self.npy_paths:
                path = os.path.join('2body_ds', _path)
                #path = os.path.join('nbody_ds3', _path)
                #path = os.path.join('.', _path)
                bodies_motion_data_list.append(np.load(path))
        except:
            print('file not found, reading from text')
        # shape is 2*4, n_steps 
        bodies_motion_data = np.stack(bodies_motion_data_list, axis=0)
        #bodies_motion_data = bodies_motion_data[0:1, :, :]
        #self.train_data = bodies_motion_data[0:6]
        #self.eval_data = bodies_motion_data[6:7]
        
        #shape B, K, N*4, n_steps
        #bodies_motion_data = bodies_motion_data[np.newaxis,...]
        #n4, n_step = bodies_motion_data.shape
        
        #use 2000 steps
        if self.train:
            #bodies_motion_data = bodies_motion_data[:,:,0:7000]
            bodies_motion_data = bodies_motion_data[:,:,0:2000]
        else:
            #bodies_motion_data = bodies_motion_data[:,:,7000:10000]
            bodies_motion_data = bodies_motion_data[:,:,2000:4000]
            
        K, n4, n_step = bodies_motion_data.shape
        #print('n step ',b,  n_step)
        self.dataset_size = (n_step-2*self.time_len)
        #self.dataset_size = (2000//self.time_len)
        
        #move time dim first
        bodies_motion_data = torch.tensor(bodies_motion_data, dtype=self.type)
        bodies_motion_data = bodies_motion_data.reshape(K, n4//4,4,n_step)
        bodies_motion_data = bodies_motion_data.permute(3,0,1,2)
        bodies_motion_data = bodies_motion_data.reshape(n_step,K,n4//4,4)
        
        #bodies_motion_data = bodies_motion_data[::10,:,:]
        #bodies_motion_data = bodies_motion_data[:2000,:,:]
        #bodies_motion_data = bodies_motion_data[:,0:10,:]
        
        self.positions = bodies_motion_data[:,:,:,0:2]
        self.velocities = bodies_motion_data[:,:,:,2:]
        
        #embed()

        mx = self.positions.mean(dim=[0], keepdim=True)
        msd = self.positions.std(dim=[0], keepdim=True)
        
        self.velocities = self.velocities
        vx = self.velocities.mean(dim=[0], keepdim=True)
        vsd = self.velocities.std(dim=[0], keepdim=True)
        
        #print('mean std', vx, vsd)
        #self.positions = self.positions#/100
        #self.positions = (self.positions -mx)/msd
        #self.positions = self.positions*10
        #self.velocities = (self.velocities -vx)/vsd#.clip(min=1e-3)
        #self.velocities = (self.velocities -vx)/vsd#.clip(min=1e-3)

    def __len__(self):
        #return self.positions.shape[0]//self.time_len
        #return (self.positions.shape[0]-self.time_len)//10
        return self.dataset_size

    def __getitem__(self, idx):
        idx = idx
        retx_in = self.positions[idx:idx+self.time_len_in]
        retx_out = self.positions[idx+self.time_len_in:idx+self.time_len_in+self.time_len]
        
        retv_in = self.velocities[idx:idx+self.time_len_in]
        retv_out = self.velocities[idx+self.time_len_in:idx+self.time_len_in+self.time_len]
        #retv = self.velocities[idx:idx+self.time_len]

        ##apply random rotaton to positions
        #if self.rotate:
        #    rot = torch.tensor(R.random().as_matrix(), dtype=self.type)
        #    retx = retx@rot#@retx.unsqueeze(2)
        #    #retx = retx.squeeze()

        return retx_in, retv_in, retx_out, retv_out, idx


class EphemerisDataModule(pl.LightningDataModule):
    def __init__(self, path, bs=1,  **kwargs):
        super().__init__()
        self.bs = bs

        self.train_dataset = EphemerisDataset(path,train=True, **kwargs)
        self.val_dataset = EphemerisDataset(path,train=False, **kwargs)
        
    #def prepare_data(self):
    #    pass

    #def setup(self, stage: Optional[str] = None):
    #    pass

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.bs, shuffle=True, drop_last=True, num_workers=12)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.bs, shuffle=False, drop_last=True)
        return val_loader

    #def test_dataloader(self):
    #    test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.bs, shuffle=False)
    #    return test_loader
