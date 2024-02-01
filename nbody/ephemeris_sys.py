

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
import numpy as np


from solver.ode_layer import ODESYSLayer

from torchmetrics.classification import Accuracy
import pytorch_lightning as pl


import pytorch_lightning as pl

from nbody.data_ephemeris import EphemerisDataModule

from scipy.spatial.transform import Rotation as R
from scipy.special import logit
import ipdb

from extras.source import write_source_files, create_log_dir

import extras.logger as logger

import tqdm
from pytorch_lightning.callbacks import TQDMProgressBar

log_dir, run_id = create_log_dir(root='logs')
write_source_files(log_dir)
L = logger.setup(log_dir)

L.info(f'run, {log_dir}, {run_id}')

PATH_DATASETS = "."  
BATCH_SIZE = 100 
DBL = True
NUM_BODIES=25
N_STEP=50



class EPHMethod(pl.LightningModule):
    def __init__(self, bs=10, data_dir=PATH_DATASETS,n_step=50, learning_rate=2e-4, **kwargs):
        super().__init__()

        self.learning_rate = 0.00001
        self.rotate =False

        self.model = EPHModel(bs=bs,n_step=n_step, **kwargs)
        self.type = torch.float64 if DBL else torch.float32
        if DBL:
            self.model = self.model.double()
            
    def setup(self, stage):
        if stage=='fit':
            print('Source files ', self.logger.log_dir)
            write_source_files(self.logger.log_dir)

    def forward(self, *args):
        return  self.model(*args)

    
    def loss(self, x, y):
        diff = (x-y).pow(2).sum(dim=-1)
        diff = (diff).mean()
        
        return diff

    def training_step(self, batch, batch_idx):

        x_in,v_in,x_out,v_out = batch[0], batch[1], batch[2], batch[3]

        x_out_model, v_out_model,eps = self(x_in, v_in)

        x_out_comb = torch.cat([x_in, x_out], dim=1)

        loss = self.loss(x_out_model, x_out_comb)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("max eps", eps, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        
        x_in,v_in,x_out,v_out = batch[0], batch[1], batch[2], batch[3]

        x_out_model, v_out_model,eps = self(x_in, v_in)

        x_out_comb = torch.cat([x_in, x_out], dim=1)
        #v_out_comb = torch.cat([v_in, v_out], dim=1)
        loss = self.loss(x_out_model, x_out_comb)

        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return optimizer
        #return ([optimizer],
        #        [{"scheduler": scheduler, "interval": "step",}]
        #       )

def generate_traj(method: EPHMethod, datamodule: EphemerisDataModule):
    method.eval()

    x_data =[]
    v_data =[]

    x_data = datamodule.val_dataset.positions
    v_data = datamodule.val_dataset.velocities

    x = x_data[:N_STEP]
    v = v_data[:N_STEP]
    x = x.unsqueeze(0).repeat(BATCH_SIZE, 1, 1, 1)
    v = v.unsqueeze(0).repeat(BATCH_SIZE, 1, 1, 1)
    #x = x0
    x = x.cuda()
    v = v.cuda()
    print(x.shape)


    out_list =[] # [x.unsqueeze(1)]
    n_step=100
    inc=25
    for i in tqdm.tqdm(range(n_step)):
        #x = true_trajectory[:, i*15:i*15+5]
        u_x,u_v, eps = method(x, v)

        u_out = u_x[0, N_STEP:inc+N_STEP]
        x = u_x[:, inc:inc+N_STEP]
        v = u_v[:, inc:inc+N_STEP]
        #x = u[:, -10:]
        out_list.append(u_out.detach().cpu())

    trajectory = torch.cat(out_list, dim=0)
    gt_trajectory = x_data
    print(trajectory.shape)

    return trajectory.detach().cpu().numpy(), gt_trajectory.cpu().numpy()


class EPHModel(nn.Module):
    def __init__(self, bs=1, order=2, n_step=50, N=2, device=None, **kwargs):
        super().__init__()

        #number of bodies
        self.N = NUM_BODIES

        self.n_step = n_step  
        self.order = 2
        # state dimension
        self.PD=3
        #self.n_dim = self.N*self.PD
        self.n_dim = self.PD
        self.n_ind_dim = self.N
        self.n_equations= self.n_dim
        self.bs = bs
        self.n_coeff = self.n_step * (self.order + 1)
        self.device = device

        #iv change to order
        self.n_iv=2 #self.order-1
        self.n_iv_steps=n_step #N_STEP

        self.step_dim = (2*self.n_step-1)*self.n_dim*self.N


        self.ode_layer = ODESYSLayer(bs=bs, n_ind_dim=self.n_ind_dim, order=self.order, n_equations=self.n_equations, 
                                     n_dim=self.n_dim, n_iv=self.n_iv, n_step=2*self.n_step, n_iv_steps=self.n_iv_steps, gamma=0.1,
                                      solver_dbl=True, double_ret=True)


        pm = 'zeros'

        self.rhs_dim = self.N*self.PD*(2*self.n_step)   

        self.n_cf_dims = self.N*self.n_equations*self.PD*(self.order+1)
        self.cf_cnn = nn.Sequential(
            nn.Conv1d(self.N*self.N*self.PD,64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(64,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(128,128, kernel_size=5, padding=2, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(128,128, kernel_size=5, padding=2, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(128,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(128,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(128,64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            #nn.Conv1d(64,7*self.coord_dims[0], kernel_size=5, padding=2),
            nn.Flatten()
            )

        self.rhs_cnn = nn.Sequential(
            nn.Conv1d(self.N*self.N*self.PD,64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(64,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(128,128, kernel_size=5, padding=2, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(128,128, kernel_size=5, padding=2, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(128,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(128,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(128,64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            #nn.Conv1d(64,7*self.coord_dims[0], kernel_size=5, padding=2),
            nn.Flatten()
            )
        
        self.coeffs_mlp = nn.Sequential(
            nn.Linear(64*50, self.N*self.n_equations*self.PD*(self.order+1)),
        )

        self.rhs_mlp = nn.Sequential(
            nn.Linear(64*50, self.N*self.n_equations),
        )

        self.steps_layer = nn.Linear(1024, self.step_dim)
        #batch, ind_dim, time, dimension

        #step_bias = logit(0.01)
        step_bias = (0.01)
        self.step_sizes = nn.Parameter(step_bias*torch.ones(1, self.N, 1, self.PD))

        self.steps_layer.weight.data.fill_(0.0)
        self.steps_layer.bias.data.fill_(step_bias)
        
        
    def forward(self, x, v):
        #(b, N_STEP, N, 3)
        pos_x, vel_x = x, v
        
        b,n_step,nb,nd = pos_x.shape
        pos_x = pos_x.reshape(b,n_step,nb,nd)
        vel_x = vel_x.reshape(b,n_step,nb,nd)

        #pairwise distances
        pairwise_dist  = pos_x.unsqueeze(2) - pos_x.unsqueeze(3)
        #pairwise_dist = pairwise_dist.pow(2).sum(dim=-1).reshape(b,N*N)
        pairwise_dist = pairwise_dist.reshape(b, n_step*self.N*self.N*self.PD)
        
        pairwise_vel  = vel_x.unsqueeze(2) - vel_x.unsqueeze(3)
        pairwise_vel = pairwise_vel.reshape(b, n_step*self.N*self.N*self.PD)
        
        
        _idin = pairwise_dist.abs()

        cin = _idin.reshape(b, n_step, self.N*self.N*self.PD).permute(0,2,1)

        _coeffs = self.cf_cnn(cin)
        coeffs = self.coeffs_mlp(_coeffs)

        coeffs = coeffs.reshape(x.shape[0], self.n_ind_dim, self.n_equations, 1,self.n_dim, self.order+1).repeat(1,1,1,2*self.n_step,1,1)

        steps = self.step_sizes.repeat(x.shape[0], 1, 2*self.n_step-1, 1).clip(min=0.0001)

        _rhs = self.rhs_cnn(cin)
        rhs = self.rhs_mlp(_rhs)
        
        rhs = rhs.reshape(b, self.n_ind_dim, self.n_equations, 1).repeat(1,1,1, 2*self.n_step)


        iv_pos =pos_x[:,:self.n_iv_steps,:,:].unsqueeze(4)
        iv_vel =vel_x[:,:self.n_iv_steps,:,:].unsqueeze(4)

        iv_rhs = torch.cat([iv_pos, iv_vel], dim=4)

        #ind dim earlier
        iv_rhs = iv_rhs.permute(0,2,1,3,4)

        #iv_rhs = iv
        u0,u1,u2,eps,steps = self.ode_layer(coeffs, rhs, iv_rhs, steps)


        #time dim earlier
        x_out = u0.permute(0,2,1,3)#.unsqueeze(2)
        v_out = u1.permute(0,2,1,3)#.unsqueeze(2)


        return x_out, v_out, eps.abs().max()


datamodule = EphemerisDataModule(path='datasets', DBL=DBL, num_bodies=NUM_BODIES, time_len=N_STEP, bs=BATCH_SIZE, spherical=False)
method = EPHMethod(bs=BATCH_SIZE, n_step=N_STEP)


trainer = pl.Trainer(
    max_epochs=500,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    check_val_every_n_epoch=1,
    callbacks=[
        pl.callbacks.ModelCheckpoint(mode="min", monitor="val_loss"),
    ],
    log_every_n_steps=500,
)
    
def train():
    trainer.fit(method, datamodule=datamodule)


def load():
    ckpt = trainer.checkpoint_callback.best_model_path
    method = EPHMethod.load_from_checkpoint(ckpt, bs=BATCH_SIZE, n_step=N_STEP)
    return method



def generate(ckpt):
    method = EPHMethod.load_from_checkpoint(ckpt, bs=BATCH_SIZE, n_step=N_STEP)
    #trainer.test(method, datamodule=datamodule)
    with torch.no_grad():
        return generate_traj(method, datamodule)


def resume(ckpt):
    #ckpt = trainer.checkpoint_callback.best_model_path
    method = EPHMethod.load_from_checkpoint(ckpt, bs=BATCH_SIZE, n_step=N_STEP)
    #return method
    trainer.fit(method, datamodule=datamodule)
    #trainer.fit()

import sys
if __name__ == "__main__":
    train()
    #generate(sys.argv[1])

    #print('open ', sys.argv[1])
    #resume(sys.argv[1])