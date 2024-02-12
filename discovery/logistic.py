
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.autograd import gradcheck

import sys

from torch.utils.data import Dataset, DataLoader
from scipy.integrate import odeint

from extras.source import write_source_files, create_log_dir

from solver.ode_layer import ODEINDLayer
import discovery.basis as B
import ipdb
import extras.logger as logger
import os

from scipy.special import logit
import torch.nn.functional as F
from tqdm import tqdm
import discovery.plot as P


log_dir, run_id = create_log_dir(root='logs')
write_source_files(log_dir)
L = logger.setup(log_dir, stdout=True)

DBL=True
dtype = torch.float64 if DBL else torch.float32
STEP = 0.05
tend = 3.0
T = int(tend/STEP)

n_step_per_batch = T
batch_size= 64

cuda=True


class LogisticDataset(Dataset):
    def __init__(self, n_step_per_batch=100, n_step=1000, train=True):
        data, shared_r, Ks, y0s = self.generate_logistic(N=5000, noise=1e-3)
        #data, shared_r, Ks, y0s = self.generate_logistic(N=5000, noise=0)

        self.down_sample = 1

        data = torch.tensor(data, dtype=dtype) 
        shared_r = torch.tensor(shared_r)
        Ks = torch.tensor(Ks)

        train_len = int(0.8*data.shape[0])
        if train:
            self.data = data[:train_len]
            self.shared_r = shared_r[:train_len]
            self.Ks = Ks[:train_len]
        else:
            self.data = data[train_len:]
            self.shared_r = shared_r[train_len:]
            self.Ks = Ks[train_len:]

        print('data shape ', self.data.shape)

    def generate_logistic(self, N=5000,noise=1e-3):
        """ dx/dt = r x - r/k x**2 """

        shared_r=np.random.randn(N)*.5 + 3.
        Ks = np.random.randn(N)*.5 + 3.
        y0s = np.random.randn(N)*.05 + 0.05
        
        #dt = 0.05
        #ts = np.linspace(0, tend, int(tend/STEP))
        ts = np.linspace(0, tend, T)
        
        nomin = Ks[:, None] * y0s[:, None] * np.exp(shared_r[:, None] * ts[None])
        denom = Ks[:,  None] + y0s[:, None] * (np.exp(shared_r[:, None] * ts[None]) - 1)
        traj = (nomin / denom) + noise*np.random.randn(len(ts))
        #return traj # (num_samples, num_timesteps)
        return traj, shared_r, Ks, y0s # (num_samples, num_timesteps)


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x =  self.data[idx]
        return x,self.shared_r[idx],self.Ks[idx]

ds = LogisticDataset(n_step=T,n_step_per_batch=n_step_per_batch, train=True)#.generate()
eval_ds = LogisticDataset(n_step=T,n_step_per_batch=n_step_per_batch, train=True)#.generate()
train_loader =DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True) 
eval_loader =DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, bs, n_step,n_step_per_batch, device=None, **kwargs):
        super().__init__()

        self.n_step = n_step #+ 1
        self.order = 3
        # state dimension
        self.bs = bs
        self.device = device
        self.n_iv=1
        self.n_ind_dim = 1
        self.n_step_per_batch = n_step_per_batch

        self.num_params = 2

        #self.step_size = nn.Parameter(logit(0.05)*torch.ones(1,1,1))
        
        self.ode = ODEINDLayer(bs=bs, order=self.order, n_ind_dim=self.n_ind_dim, n_step=self.n_step_per_batch, solver_dbl=True, double_ret=True,
                                    n_iv=self.n_iv, n_iv_steps=1,  gamma=0.05, alpha=0, **kwargs)


        pm = 'zeros'
        self.cf_cnn = nn.Sequential(
            nn.Conv1d(1,64, kernel_size=3, padding=1, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(64,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(128,256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(256,256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(256,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(128,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(128,1, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            )

        self.param_cnn = nn.Sequential(
            nn.Conv1d(1,64, kernel_size=3, padding=1, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(64,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(128,256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(256,256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(256,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(128,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.Flatten(),
            nn.Linear(60*128,2)
            )

        self.step_cnn = nn.Sequential(
            nn.Conv1d(1,64, kernel_size=3, padding=1, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(64,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(128,256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(256,256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(256,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(128,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.Flatten(),
            )
        self.step_mlp = nn.Linear(60*128,1)

        step_bias_t = logit(0.05)
        self.step_mlp.weight.data.fill_(0.0)
        self.step_mlp.bias.data.fill_(step_bias_t)
    

    def forward(self, x):
        # apply mask
        var = self.cf_cnn(x.reshape(self.bs,-1).unsqueeze(1))
        var = var.reshape(self.bs, self.n_step_per_batch)

        params = self.param_cnn(x.reshape(self.bs,-1).unsqueeze(1))
        #params = self.param_cnn(var.reshape(self.bs,-1).unsqueeze(1))
        params = params.reshape(-1, 2, 1)

        rhs = params[:,0]*var + params[:,1]*(var**2)

        z = torch.zeros(1, self.n_ind_dim, 1,1).type_as(x)
        o = torch.ones(1, self.n_ind_dim, 1,1).type_as(x)

        # 0*u + 1*u' + 0*u'' +0u''' = rhs
        coeffs = torch.cat([z,o,z,z], dim=-1)
        coeffs = coeffs.repeat(self.bs,1,self.n_step_per_batch,1)

        init_iv = var[:,0]

        #steps = self.step_size.repeat(self.bs, self.n_ind_dim, self.n_step_per_batch-1).type_as(x)
        steps = self.step_cnn(x.reshape(self.bs,-1).unsqueeze(1))
        steps = self.step_mlp(steps).reshape(self.bs,1,1)
        #steps = self.step_size.repeat(self.bs, self.n_ind_dim, self.n_step_per_batch-1).type_as(x)
        steps = steps.repeat(1, self.n_ind_dim, self.n_step_per_batch-1).type_as(x)

        steps = torch.sigmoid(steps).clip(min=0.0005)
        #self.steps = self.steps.type_as(net_iv)

        x0,x1,x2,eps,steps = self.ode(coeffs, rhs, init_iv, steps)
        x0 = x0.permute(0,2,1)
        #ipdb.set_trace()

        x0 = x0.squeeze()

        return x0, steps, eps.abs().max(), var,params

model = Model(bs=batch_size,n_step=T, n_step_per_batch=n_step_per_batch, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

if DBL:
    model = model.double()
model=model.to(device)



def loss_func(x, y):
    #MAE loss
    loss = (x- y).abs().mean()
    #loss = (x- y).pow(2).mean()
    return loss


def train_step(batch):
    model.train()
    batch = batch.to(device)

    optimizer.zero_grad()
    x0, steps, eps, var,xi = model(batch)

    x_loss = loss_func(x0, batch)
    v_loss = loss_func(var, x0)
    loss = x_loss +  v_loss
    

    loss.backward()
    optimizer.step()

    return {'loss': loss, 'x_loss':x_loss, 'v_loss':v_loss,
            'x0': x0, 'eps': eps, 'var':var, 'xi': xi}

def eval_step(batch):
    model.eval()
    batch = batch.to(device)

    x0, steps, eps, var,xi = model(batch)

    x_loss = loss_func(x0, batch)
    v_loss = loss_func(var, x0)
    loss = x_loss +  v_loss
    
    return {'loss': loss, 'x_loss':x_loss, 'v_loss':v_loss,
            'x0': x0, 'eps': eps, 'var':var, 'xi': xi}

def evaluate():
    losses = []
    for i, (batch,_,_) in enumerate(eval_loader):
        batch = batch.to(device)
        ret = eval_step(batch)
        losses.append(ret['loss'].item())
    eval_loss = np.array(losses).mean()
    L.info(f'eval loss {eval_loss}')
    return eval_loss

def check(label):
    learned_params = []
    act_rs = []
    act_rks = []
    #check parameters
    for i, (batch,rs,ks) in enumerate(eval_loader):
        batch = batch.to(device)
        params = model.param_cnn(batch.reshape(batch_size,-1).unsqueeze(1))
        params = params.squeeze()

        act_rs.append(rs)
        act_rks.append(-rs/ks)
        learned_params.append(params)
        #load debugger for manual check
        #ipdb.set_trace()
    learned_params = torch.cat(learned_params, dim=0)
    act_rs = torch.cat(act_rs, dim=0)
    act_rks = torch.cat(act_rks, dim=0)

    learned_params = learned_params.detach().cpu().numpy()
    act_rs = act_rs.detach().cpu().numpy()
    act_rks = act_rks.detach().cpu().numpy()

    P.plot_logistic(learned_params, act_rs, act_rks, os.path.join(log_dir, f'params_{label}_all.pdf'))
    P.plot_logistic(learned_params[:100], act_rs[:100], act_rks[:100], os.path.join(log_dir, f'params_{label}_100.pdf'))

def train(nepoch=300):
    with tqdm(total=nepoch) as pbar:
        eval_loss = 9999
        for epoch in range(nepoch):
            pbar.update(1)
            losses = []
            xlosses = []
            vlosses = []
            for i, (batch,_,_) in enumerate(train_loader):
                batch = batch.to(device)
                ret = train_step(batch)

                losses.append(ret['loss'].item())
                xlosses.append(ret['x_loss'].item())
                vlosses.append(ret['v_loss'].item())

            train_loss = np.array(losses).mean()
            x_loss = np.array(xlosses).mean()
            v_loss = np.array(vlosses).mean()

            xi = ret['xi']
            eps = ret['eps']

            xi = xi.detach().cpu().numpy()
            meps = eps.max().item()
            #L.info(f'run {run_id} epoch {epoch}, loss {train_loss} max eps {meps} xloss {x_loss} vloss {v_loss} ')
            pbar.set_description(f'run {run_id} epoch {epoch}, loss {train_loss} max eps {meps} xloss {x_loss} vloss {v_loss} ')

            if epoch%10 == 0:
                _eval_loss = evaluate()
                if _eval_loss < eval_loss:
                    eval_loss = _eval_loss
                    torch.save(model.state_dict(), os.path.join(log_dir, 'model.ckpt'))
                #plot params
                check(epoch)

def load(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path))
    evaluate()
    check(label='test')


if __name__ == "__main__":
    train()

    #load(sys.argv[1])
    #print_eq()
