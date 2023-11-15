
import sys
sys.path.append("../../")

import torch.nn as nn
import torch
# import lp_system
from torch.nn.parameter import Parameter
import numpy as np

#from lp_dyn import ODELP
#from lp_dyn_cent import ODELP
import matplotlib.pyplot as plt
#import osqp
import torch

from solver.ode_layer import ODESYSLayer
#from qp_primal_direct_batched import QPFunction
# from qp_qp import QPFunction
import torch.optim as optim
from matplotlib.animation import FuncAnimation
from torch.autograd import gradcheck

from torchmetrics.classification import Accuracy
import pytorch_lightning as pl

from torch.utils.data import Dataset

import os

import pytorch_lightning as pl
# import pandas as pd
# import seaborn as sn
import torch
# from IPython.display import display
# from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
from nbody.data_multi import EphemerisDataModule
from IPython import embed
#from lightning.pytorch.callbacks import TQDMProgressBar

from scipy.spatial.transform import Rotation as R
from scipy.special import logit
import ipdb

from extras.source import write_source_files, create_log_dir

import extras.logger as logger

log_dir, run_id = create_log_dir(root='logs')
write_source_files(log_dir)
L = logger.setup(log_dir)

L.info(f'run, {log_dir}, {run_id}')

DBL = False
NUM_BODIES=2
N_STEP=50


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
        self.rotate =False

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
        b, T, N, _ = x.shape
        init_x = x[:, 0, :, :]
        init_v = v[:, 0, :, :]
        return init_x, init_v
    
    def loss(self, x, y):
        #b, t, n, 2
        diff = (x-y).pow(2).sum(dim=-1)
        #d = (y).pow(2).sum(dim=-1)
        
        #diff = (diff/d).mean()
        diff = (diff).mean()
        
        return diff

    def training_step(self, batch, batch_idx):

        x_in,v_in,x_out,v_out = batch[0], batch[1], batch[2], batch[3]

        #shift to 0
        x_init = x_in[:,0:1]#.mean(dim=3,keepdim=True)
        x_in = x_in - x_init
        x_out = x_out - x_init

        #batch = (x_in, v_in)
        x_out_model, v_out_model = self((x_in, v_in))

        x_out_comb = torch.cat([x_in, x_out], dim=1)

        loss = self.loss(x_out_model, x_out_comb)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        
        x_in,v_in,x_out,v_out = batch[0], batch[1], batch[2], batch[3]

        #shift to 0
        x_init = x_in[:,0:1]#.mean(dim=3,keepdim=True)
        x_in = x_in - x_init
        x_out = x_out - x_init

        x_out_model, v_out_model = self((x_in, v_in))

        x_out_comb = torch.cat([x_in, x_out], dim=1)
        loss = self.loss(x_out_model, x_out_comb)

        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        
    def inference(self, batch):
        self.eval()

        x_in,v_in = batch[0], batch[1]

        #shift to 0
        x_init = x_in[:,0:1]#.mean(dim=3,keepdim=True)
        x_in = x_in - x_init

        #x,v = self.model(x)
        x,v = self((x_in, v_in))

        #shift back
        x = x+x_init
        return x,v

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        #
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
        
        return optimizer
        #return ([optimizer],
        #        [{"scheduler": scheduler, "interval": "step",}]
        #       )


class EPHModel(nn.Module):
    def __init__(self, bs=1, order=2, n_step=50, N=2, solver_dbl=True, device=None, **kwargs):
        super().__init__()

        #number of bodies
        self.N = NUM_BODIES
        # self.step_size = 0.1
        # self.end = 30 * self.step_size
        self.n_step = n_step  # int(self.end /self.step_size)
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
        self.n_iv_steps=N_STEP

        self.step_dim = (2*self.n_step-1)*self.n_dim*self.N
        #self.step_dim = self.n_dim

        self.ode_layer = ODESYSLayer(bs=bs, n_ind_dim=self.n_ind_dim, order=self.order, n_equations=self.n_equations,  gamma=0.5, alpha=0.,
                                     n_dim=self.n_dim, n_iv=self.n_iv, n_step=2*self.n_step, n_iv_steps=self.n_iv_steps, solver_dbl=solver_dbl)


        #coeffs = coeffs.reshape(self.bs*self.n_ind_dim, self.n_equations, self.n_step,self.n_dim, self.order + 1)
        _coeffs = torch.tensor(np.random.random((1, self.n_ind_dim, self.n_dim, 1, self.n_dim, self.order+1)))
        self.coeffs = Parameter(_coeffs)

        #self.gru = GRULayer(in_dim=self.N*2, out_dim=self.N*self.PD*(self.order+1))
        self.rhs_dim = self.N*self.PD*(2*self.n_step)   
        self.rhs_t = nn.Sequential(
            nn.Linear((self.N*self.N)*self.PD*self.n_step, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.N*self.n_equations*(2*self.n_step)),
        )
        
        self.pre_coeffs_mlp = nn.Sequential(
            nn.Linear((self.N*self.N)*self.PD*self.n_step, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        
        self.pre_steps_mlp = nn.Sequential(
            #nn.Linear(self.N*self.PD, 2048),
            #nn.Linear((self.N*self.N+self.N*self.N)*self.PD*self.n_step, 2048),
            #nn.Linear((2*self.N*self.N)*self.PD*self.n_step, 1024),
            nn.Linear((self.N*self.N)*self.PD*self.n_step, 1024),
            #nn.Linear((self.N+self.N*self.N)*self.PD*self.n_step, 1024),
            #nn.Linear((self.N+self.N)*self.PD*self.n_step, 2048),
            #nn.Linear(self.N*self.N*self.PD*self.n_step, 2048),
            #nn.Linear(2*self.N*self.PD , 2048),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )

        self.coeffs_mlp = nn.Sequential(
            nn.Linear(1024, self.N*self.n_equations*self.PD*(2*self.n_step)*(self.order+1)),
        )

        self.steps_layer = nn.Linear(1024, self.step_dim)

        step_bias = logit(0.1)
        self.steps_layer.weight.data.fill_(0.0)
        self.steps_layer.bias.data.fill_(step_bias)
        
        self.out_mlp = nn.Sequential(
            nn.Linear(self.N*self.PD, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.N*self.PD)
            #nn.Linear(2048, self.n_step*(self.order+1)),
        )
        
            
        
    def pw(self, f):
        b, _ = f.shape
        f = f.reshape(b*self.PD*(2*self.n_step), self.N, self.N)
        f = torch.tril(f, diagonal=-1) 
        ft = f.permute(0,2,1)
        f = f-ft
        f = f.sum(dim=-1)
        f = f.reshape(b,self.PD,2*self.n_step, self.N)
        f = f.permute(0,3,1,2)
        return f
    
    def get_f(self, idin):
        
        outs = []
        for i in range(self.N):
            out = self.fs[i](idin)
            outs.append(out)
        outs = torch.stack(outs, dim=1)
        
        outs = outs.reshape(-1, self.N*self.PD*self.n_step)
        return outs

    def forward(self, x):
        #if DBL:
        #    x = x.double()

        
        #(b, N_STEP, K, 2, 2)
        pos_x, vel_x = x[0], x[1]
        #ix = self.init_fp(x[0])
        #iv = self.init_fp(x[1])

        
        pos_x = pos_x.permute(0,2,1,3,4)
        vel_x = vel_x.permute(0,2,1,3,4)
        
        b,k,n_step,nb,nd = pos_x.shape
        pos_x = pos_x.reshape(b*k,n_step,nb,nd)
        vel_x = vel_x.reshape(b*k,n_step,nb,nd)

        #pairwise distances
        pairwise_dist  = pos_x.unsqueeze(2) - pos_x.unsqueeze(3)
        #pairwise_dist = pairwise_dist.pow(2).sum(dim=-1).reshape(b,N*N)
        pairwise_dist = pairwise_dist.reshape(b*k, n_step*self.N*self.N*2)
        
        pairwise_vel  = vel_x.unsqueeze(2) - vel_x.unsqueeze(3)
        pairwise_vel = pairwise_vel.reshape(b*k, n_step*self.N*self.N*2)
        
        idin = pairwise_dist
        
        idin_vel = vel_x.reshape(b*k, n_step*nb*nd)
        idin_pos = pos_x.reshape(b*k, n_step*nb*nd)
        
        #_idin = pairwise_dist
        #_idin = idin_pos
        
        #embed()
        
        #_idin = torch.cat([idin_vel, pairwise_dist], dim=-1)
        #_idin = torch.cat([pairwise_vel, pairwise_dist], dim=-1)
        #_idin = torch.cat([idin_vel, idin_pos], dim=-1)
        #_idin = idin_pos
        _idin = pairwise_dist
        rhs = self.rhs_t(_idin)
        #rhs = self.rhs_t(idin_pos)
        #rhs = self.rhs_pw(_idin)
        #rhs = self.pw(rhs)
        
        #iv_rhs = self.iv_pw(pairwise_dist)
        
        #_coeffs = self.pre_coeffs_mlp(_idin)
        _steps = self.pre_steps_mlp(_idin)
        #coeffs = self.coeffs_mlp(_coeffs)
        steps = self.steps_layer(_steps)
        
        steps = torch.sigmoid(steps)#.clip(min=0.001, max=2)

        #coeffs = coeffs.reshape(-1, self.order+1)
        #coeffs[:,-1] = 0.
        #steps = 0.1*torch.ones_like(steps)
        #steps = steps.reshape(b, self.n_dim,1).repeat(1,1,2*self.n_step-1)
        #coeffs = coeffs.reshape(self.bs , self.n_dim, self.n_step, self.order + 1)
        #coeffs = coeffs.reshape(self.bs , 1, self.n_step, self.order + 1)
        #coeffs = coeffs.repeat(1, self.n_dim, 1,1)
        
        rhs = rhs.reshape(b*k, self.n_ind_dim, self.n_equations, 2*self.n_step)
        #iv_rhs = iv_rhs.reshape(b*k, self.n_dim, 2)
        #coeffs = self.gru(idin_pos)
        #rhs = self.gru(idin)
        #rhs = self.rhs_skip(idin)
        
        #rhs = self.get_f(idin)


        coeffs = self.coeffs.reshape(1, self.n_ind_dim, self.n_dim, 1, self.n_dim, self.order+1)
        coeffs = coeffs.repeat(self.bs, 1, 1, 2*self.n_step, 1, 1)

        #coeffs = coeffs.reshape(-1,self.N,self.n_equations,self.PD,(2*self.n_step),(self.order+1))
        #coeffs[:,:,:,:,:,-1] = 0.

        #set initial values
        #iv_rhs =torch.cat([pos_x[:,-1,:,:].unsqueeze(3), vel_x[:,-1,:,:].unsqueeze(3)], dim=-1)
        #iv_rhs =torch.cat([pos_x[:,0,:,:].unsqueeze(3), vel_x[:,0,:,:].unsqueeze(3)], dim=-1)
        #iv_rhs =torch.cat([pos_x[:,0,:,:].unsqueeze(3), vel_x[:,0,:,:].unsqueeze(3)], dim=-1)
        #iv_rhs =torch.cat([pos_x[:,:self.n_iv_steps,:,:].unsqueeze(4), vel_x[:,:self.n_iv_steps,:,:].unsqueeze(4)], dim=-1)
        pos_x = pos_x.squeeze()
        iv_rhs =pos_x[:,:self.n_iv_steps,:,:]#.unsqueeze(4)#, vel_x[:,:self.n_iv_steps,:,:].unsqueeze(4)], dim=-1)
        iv_rhs = iv_rhs.permute(0,2,1,3)
        #iv_rhs =pos_x[:,0,:,:]#.unsqueeze(3), vel_x[:,0,:,:].unsqueeze(3)], dim=-1)
        #iv_rhs = None
        

        #x_out, v_out, a_out = self.ode_layer(coeffs, rhs, iv_rhs)

        u0,u1,u2,eps,steps = self.ode_layer(coeffs, rhs, iv_rhs, steps)


        #shape(self.bs,self.n_ind_dim, self.n_step, self.n_dim)
        #u0 = u0.reshape(self.bs, self.n_dim//self.PD,self.PD, 2*self.n_step)
        #u1 = u1.reshape(self.bs, self.n_dim//self.PD,self.PD, 2*self.n_step)
        #b, step, nbodies, dim
        x_out = u0.permute(0,2,1,3).unsqueeze(2)
        v_out = u1.permute(0,2,1,3).unsqueeze(2)

        #if self.training:
        #    #x_out, v_out, a_out = self.l0_train(None, rhs, iv_rhs)
        #    x_out, v_out, a_out = self.l0_train(coeffs, rhs, iv_rhs)
        #else:
        #    #x_out, v_out, a_out = self.l0_eval(None, rhs, iv_rhs)
        #    x_out, v_out, a_out = self.l0_eval(coeffs, rhs, iv_rhs)
        #b*k,step,N,D
        
        #x_out = x_out.reshape(b,k,2*n_step, nb*nd)
        #x_out = self.out_mlp(x_out)
        
        #x_out = x_out.reshape(b,k,2*n_step, nb,nd).permute(0,2,1,3,4)
        #v_out = v_out.reshape(b,k,2*n_step, nb,nd).permute(0,2,1,3,4)
        
        #x_out = x_out[:,1:2*n_step]
        #v_out = v_out[:,1:2*n_step]

        return x_out, v_out#, a_out#, coeffs, rhs

#datamodule = EphemerisDataModule(PATH_DATASETS, DBL=DBL, time_len=N_STEP, bs=BATCH_SIZE, spherical=False)
#train_len = len(datamodule.train_dataset) + len(datamodule.val_dataset)
#method = EPHMethod(bs=BATCH_SIZE, n_step=N_STEP)


trainer = pl.Trainer(
    max_epochs=200,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    #accelerator="cpu",
    devices=1,
    #gradient_clip_val=0.5,
    #gradient_clip_algorithm="value",
    #limit_val_batches=0.0,
    check_val_every_n_epoch=10,
    callbacks=[
        #pl.callbacks.ModelCheckpoint(mode="min", monitor="val_loss"),
        pl.callbacks.ModelCheckpoint(mode="min"),
        #pl.callbacks.TQDMProgressBar(refresh_rate=500)#
        bar
    ],
    log_every_n_steps=500,
)

def build_model(batch_size=30, solver_dbl=True, learning_rate=0.0001):
    path_data = "." 

    datamodule = EphemerisDataModule(path_data, DBL=DBL, time_len=N_STEP, bs=batch_size, spherical=False)
    method = EPHMethod(bs=batch_size, n_step=N_STEP, learning_rate=learning_rate, solver_dbl=solver_dbl)

    return method, datamodule
    
def train(method, datamodule):
    trainer.fit(method, datamodule=datamodule)


def load(batch_size=30, solver_dbl=True, learning_rate=0.0001):
    ckpt = trainer.checkpoint_callback.best_model_path
    method = EPHMethod.load_from_checkpoint(ckpt, bs=batch_size, n_step=N_STEP, learning_rate=learning_rate, solver_dbl=solver_dbl)
    return method

if __name__ == "__main__":
    batch_size=30
    solver_dbl=True
    learning_rate=0.00005

    method, datamodule = build_model(batch_size, solver_dbl, learning_rate)
    train(method, datamodule)