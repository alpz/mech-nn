
import sys

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import numpy as np

import matplotlib.pyplot as plt
import torch

from solver.ode_layer import ODESYSLayer
import torch.optim as optim
import pytorch_lightning as pl

import os

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from nbody.data_multi import EphemerisDataModule
from scipy.special import logit
import ipdb

from extras.source import write_source_files, create_log_dir

import extras.logger as logger

log_dir, run_id = create_log_dir(root='logs')
write_source_files(log_dir)
L = logger.setup(log_dir)

L.info(f'run, {log_dir}, {run_id}')

#PATH_DATASETS = "."  # os.environ.get("PATH_DATASETS", ".")
#BATCH_SIZE = 30 # if torch.cuda.is_available() else 5
DBL = False
#NUM_BODIES=4
NUM_BODIES=2
N_STEP=50

#LEARNING_RATE = 0.0001

import tqdm
from pytorch_lightning.callbacks import TQDMProgressBar

class LitProgressBar(TQDMProgressBar):
   
    def init_validation_tqdm(self):
        bar = tqdm.tqdm(            
            disable=True,            
        )
        return bar

bar = LitProgressBar()


class EPHMethod(pl.LightningModule):
    def __init__(self, bs=10, n_step=50, learning_rate=0.0001, **kwargs):
        super().__init__()

        self.learning_rate = learning_rate

        self.model = EPHModel(bs=bs,n_step=n_step, **kwargs)
        self.type = torch.float64 if DBL else torch.float32
        if DBL:
            self.model = self.model.double()
            

    def forward(self, x):
        #x_out, v_out, a_out, coeffs, rhs = self.model(x)
        x_out, v_out= self.model(x)
        return x_out, v_out#, a_out, coeffs, rhs

    def get_init_vals(self, batch):
        x, v = batch[0], batch[1]
        init_x = x[:, 0, :, :]
        init_v = v[:, 0, :, :]
        return init_x, init_v
    
    def loss(self, x, y):
        #b, t, n, 2
        diff = (x-y).pow(2).sum(dim=-1)
        diff = diff.mean()
        
        return diff

    def training_step(self, batch, batch_idx):

        x_in,v_in,x_out,v_out = batch[0], batch[1], batch[2], batch[3]

        #shift to 0. Subtracts 0th step value. Trains better.
        x_init = x_in[:,0:1]
        x_in = x_in - x_init
        x_out = x_out - x_init

        x_out_model, v_out_model = self((x_in, v_in))

        x_out_comb = torch.cat([x_in, x_out], dim=1)
        #v_out_comb = torch.cat([v_in, v_out], dim=1)

        loss = self.loss(x_out_model, x_out_comb)
        #loss = loss+F.mse_loss(v_out, batch[1])
        #loss = loss+self.loss(v_out_model, v_out_comb)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        #print(batch_idx, loss)
        return loss

    def validation_step(self, batch, batch_idx):
        
        x_in,v_in,x_out,v_out = batch[0], batch[1], batch[2], batch[3]

        #shift to 0
        x_init = x_in[:,0:1]
        x_in = x_in - x_init
        x_out = x_out - x_init

        x_out_model, v_out_model = self((x_in, v_in))

        x_out_comb = torch.cat([x_in, x_out], dim=1)
        #v_out_comb = torch.cat([v_in, v_out], dim=1)
        loss = self.loss(x_out_model, x_out_comb)
        #loss = loss+self.loss(v_out_model, v_out_comb)

        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        
    def inference(self, batch):
        self.eval()

        x_in,v_in = batch[0], batch[1]

        #shift to 0
        x_init = x_in[:,0:1]
        x_in = x_in - x_init

        #x,v = self.model(x)
        x,v = self((x_in, v_in))

        #shift back
        x = x+x_init
        return x,v

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
        return optimizer
        #return ([optimizer],
        #        [{"scheduler": scheduler, "interval": "step",}]
        #       )


class EPHModel(nn.Module):
    def __init__(self, bs=1, order=2, n_step=50, N=2, solver_dbl=True, device=None, **kwargs):
        super().__init__()

        #number of bodies
        self.N = NUM_BODIES

        self.n_step = n_step  
        self.order = 2
        # state dimension
        self.PD=2
        #self.n_dim = self.N*self.PD
        self.n_dim = self.PD
        self.n_ind_dim = self.N
        self.n_equations= self.n_dim
        self.bs = bs
        self.n_coeff = self.n_step * (self.order + 1)
        self.device = device
        self.n_iv=1
        self.n_iv_steps=n_step

        self.step_dim = (2*self.n_step-1)*self.n_dim*self.N

        self.ode_layer = ODESYSLayer(bs=bs, n_ind_dim=self.n_ind_dim, order=self.order, n_equations=self.n_equations, 
                                     n_dim=self.n_dim, n_iv=self.n_iv, n_step=2*self.n_step, n_iv_steps=self.n_iv_steps, solver_dbl=solver_dbl)


        self.rhs_dim = self.N*self.PD*(2*self.n_step)   
        self.rhs_t = nn.Sequential(
            nn.Linear((self.N*self.N)*self.PD*self.n_step, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            #
            nn.Linear(2048, 1024),
            nn.ReLU(),
            
            nn.Linear(1024, self.N*self.n_equations*(2*self.n_step)),
        )
        
        self.coeff_param = nn.Parameter(torch.rand(1,self.N, 1024))

        self.coeffs_mlp = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            #nn.Linear(1024, self.N*self.n_equations*self.PD*(self.order+1)),
            nn.Linear(1024, self.n_equations*self.PD*(self.order+1)),
        )
        
        self.pre_steps_mlp = nn.Sequential(
            nn.Linear((self.N*self.N)*self.PD*self.n_step, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )

        self.steps_layer = nn.Linear(1024, self.step_dim)

        #set step bias to make initial step 0.1
        step_bias = logit(0.1)
        self.steps_layer.weight.data.fill_(0.0)
        self.steps_layer.bias.data.fill_(step_bias)

    def forward(self, x):
        """
            Given N_STEP trajectory, produces ODE for 2*N_STEP trajectory where the first
            N_STEPs are set to the input as initial values. 
        """
        #if DBL:
        #    x = x.double()

        
        #(b, N_STEP, K, 2, 2)
        pos_x, vel_x = x[0], x[1]

        
        pos_x = pos_x.permute(0,2,1,3,4)
        vel_x = vel_x.permute(0,2,1,3,4)
        
        b,k,n_step,nb,nd = pos_x.shape
        pos_x = pos_x.reshape(b*k,n_step,nb,nd)
        vel_x = vel_x.reshape(b*k,n_step,nb,nd)

        #pairwise distances
        pairwise_dist  = pos_x.unsqueeze(2) - pos_x.unsqueeze(3)
        pairwise_dist = pairwise_dist.reshape(b*k, n_step*self.N*self.N*2)
        
        rhs = self.rhs_t(pairwise_dist)
        
        #Time varying ODE coefficients
        coeffs_in = self.coeff_param.repeat(self.bs,1,1)
        coeffs = self.coeffs_mlp(coeffs_in)

        #expand coeffs over time
        coeffs = coeffs.reshape(self.bs*self.n_ind_dim,self.n_equations, 1,self.n_dim*(self.order + 1)).repeat(1,1,2*self.n_step,1)

        #Learned steps
        _steps = self.pre_steps_mlp(pairwise_dist)
        steps = self.steps_layer(_steps)
        
        steps = torch.sigmoid(steps).clip(min=0.001, max=0.999)

        
        #ODE rhs. Time varying
        rhs = rhs.reshape(b*k, self.n_ind_dim, self.n_equations, 2*self.n_step)

        #Set initial values for the first N_STEPs. Could also be learned.
        pos_x = pos_x.squeeze()
        iv_rhs =pos_x[:,:self.n_iv_steps,:,:]
        iv_rhs = iv_rhs.permute(0,2,1,3)

        #Solve ODE. 
        u0,u1,u2,eps,steps = self.ode_layer(coeffs, rhs, iv_rhs, steps)


        x_out = u0.permute(0,2,1,3).unsqueeze(2)
        v_out = u1.permute(0,2,1,3).unsqueeze(2)


        return x_out, v_out


def build_model(batch_size=30, solver_dbl=True, learning_rate=0.0001):

    path_data = "." 

    datamodule = EphemerisDataModule(path_data, DBL=DBL, time_len=N_STEP, bs=batch_size, spherical=False)
    method = EPHMethod(bs=batch_size, n_step=N_STEP, learning_rate=learning_rate, solver_dbl=solver_dbl)

    return method, datamodule


trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    #accelerator="cpu",
    devices=1,
    #limit_val_batches=0.0,
    check_val_every_n_epoch=10,
    callbacks=[
        pl.callbacks.ModelCheckpoint(mode="min"),
        #pl.callbacks.TQDMProgressBar(refresh_rate=500)#
        bar
    ],
    log_every_n_steps=500,
)
    
def train(method, datamodule):
    trainer.fit(method, datamodule=datamodule)


def load(batch_size=30, solver_dbl=True, learning_rate=0.0001):
    ckpt = trainer.checkpoint_callback.best_model_path
    method = EPHMethod.load_from_checkpoint(ckpt, bs=batch_size, n_step=N_STEP, learning_rate=learning_rate, solver_dbl=solver_dbl)
    return method

if __name__ == "__main__":
    batch_size=30
    solver_dbl=True
    learning_rate=0.0001

    method, datamodule = build_model(batch_size, solver_dbl, learning_rate)
    train(method, datamodule)