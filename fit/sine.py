
import torch.nn as nn
import torch

from solver.ode_layer import ODEINDLayer
from torch.nn.parameter import Parameter
import numpy as np

import torch
import ipdb
import torch.optim as optim
import pytorch_lightning as pl

from torch.utils.data import Dataset

#Fit a noisy exponentially damped sine wave with a second order ODE

class SineDataset(Dataset):
    def __init__(self, end=10, n_step=50):
        _y = np.linspace(0, end, n_step)
        y0 = np.sin(2*_y) + 0.5*np.random.randn(*_y.shape)
        y1 = np.cos(_y) #+ 0.5*np.random.randn(*_y.shape)
        y0 = torch.tensor(y0)

        damp = np.exp(-0.1*_y)
        self.y = y0*damp 
        
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        y = self.y #+ 0.5*torch.randn_like(self.y)
        return y
    
class SineDataModule(pl.LightningDataModule):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=True)
        return train_loader 
        
        

class Method(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.01
        self.model = Sine(device=self.device)
        
        self.func_list = []
        self.y_list = []
        self.funcp_list = []
        self.funcpp_list = []
        self.steps_list = []

    def forward(self):
        return self.model(check=False)
    

    def training_step(self, batch, batch_idx):
        y = batch
        eps, u0, u1,u2,steps = self()
        
        loss = (u0-y).pow(2).sum()
        
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('eps', eps, prog_bar=True, logger=True)
        
        self.func_list.append(u0.detach().cpu().numpy())
        self.funcp_list.append(u1.detach().cpu().numpy())
        self.funcpp_list.append(u2.detach().cpu().numpy())
        self.steps_list.append(steps.detach().cpu().numpy())
        
        self.y_list.append(y.detach().cpu().numpy())
        
        return {"loss": loss, 'y':y, 'u0':u0, 'u1':u1,'u2': u2}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer


class Sine(nn.Module):
    def __init__(self, bs=1, device=None):
        super().__init__()

        self.step_size = 0.1
        self.end = 500* self.step_size
        self.n_step = int(self.end /self.step_size)
        self.order = 2
        #state dimension
        self.n_dim = 1
        self.bs = bs
        self.n_coeff = self.n_step * (self.order + 1)
        self.device  = device
        dtype = torch.float64

        _coeffs = torch.tensor(np.random.random((self.n_dim, self.n_step, self.order+1)), dtype=dtype)
        self.coeffs = Parameter(_coeffs)


        #initial values grad and up
        self.n_iv = 0
        if self.n_iv > 0:
            iv_rhs = np.array([0]*self.n_dim).reshape(self.n_dim,self.n_iv)
            iv_rhs = torch.tensor(iv_rhs, dtype=dtype)
            self.iv_rhs = Parameter(iv_rhs)
        else:
            self.iv_rhs = None

        _rhs = np.array([1] * self.n_step)
        _rhs = torch.tensor(_rhs, dtype=dtype, device=self.device).reshape(1,1,-1).repeat(self.bs, self.n_dim,1)
        self.rhs = _rhs

        self.steps = torch.logit(self.step_size*torch.ones(1,self.n_step-1,self.n_dim))
        self.steps = nn.Parameter(self.steps)

        self.ode = ODEINDLayer(bs=bs, order=self.order, n_ind_dim=self.n_dim, n_iv=self.n_iv, n_step=self.n_step, n_iv_steps=1)

        
    def forward(self, check=False):
        coeffs = self.coeffs.unsqueeze(0).repeat(self.bs,1,1,1)

        iv_rhs = self.iv_rhs
        
        if self.n_iv > 0:
            iv_rhs = iv_rhs.unsqueeze(0).repeat(self.bs,1, 1)

        steps = torch.sigmoid(self.steps)
        steps = steps.repeat(self.bs,1, 1).double()

        rhs = self.rhs.type_as(coeffs)

        u0,u1,u2,eps,steps = self.ode(coeffs, rhs, iv_rhs, steps)


        return eps, u0, u1,u2,steps

method = Method()

def train():
    dataset = SineDataset(end=method.model.end, n_step=method.model.n_step)
    datamodule = SineDataModule(dataset=dataset)

    trainer = pl.Trainer(
        max_epochs=500,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        #accelerator="cpu",
        devices=1,
        callbacks=[
            #pl.callbacks.ModelCheckpoint(mode="max", monitor="val_accuracy"),
            #pl.callbacks.RichProgressBar(),
        ],
        log_every_n_steps=1,
    )
    trainer.fit(method, datamodule=datamodule)


if __name__ == "__main__":
    train()