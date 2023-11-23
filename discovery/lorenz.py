
#import sys
#sys.path.append("../../")

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from matplotlib.animation import FuncAnimation
from torch.autograd import gradcheck

from torchmetrics.classification import Accuracy
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from scipy.integrate import odeint

from extras.source import write_source_files, create_log_dir

from solver.ode_layer import ODEINDLayer#, ODESYSLayer
#from ode import ODEForwardINDLayer#, ODESYSLayer
import discovery.basis as B
import ipdb
import extras.logger as logger

#gradcheck = torch.sparse.as_sparse_gradcheck(torch.autograd.gradcheck)

log_dir, run_id = create_log_dir(root='logs')
write_source_files(log_dir)
L = logger.setup(log_dir)

#N = 2
DBL=True
dtype = torch.float64 if DBL else torch.float32
STEP = 0.001
cuda=True
T = 80000
n_step_per_batch = 50
batch_size= 20
threshold = 0.5

class LorenzDataset(Dataset):
    def __init__(self, n_step_per_batch=100, n_step=1000):
        #_y = np.linspace(0, end, n_step)
        self.n_step_per_batch=n_step_per_batch
        self.n_step=n_step
        self.end= n_step*STEP
        #print('step size ', end/n_step)
        x_train = self.generate()

        print('creating basis')
        basis,basis_vars =B.create_library(x_train, polynomial_order=2, use_trig=False, constant=False)
        print('basis shape', basis.shape)

        self.x_train = torch.tensor(x_train) 
        self.x_train = self.x_train #+ 0.5*torch.randn_like(self.x_train)
        print('x train shape ', self.x_train.shape)
        self.basis = torch.tensor(basis)
        self.basis_vars = basis_vars
        self.n_basis = self.basis.shape[1]

    def generate(self):
        rho = 28.0
        sigma = 10.0
        beta = 8.0 / 3.0
        dt = 0.01

        def f(state, t):
            x, y, z = state
            return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

        state0 = [1.0, 1.0, 1.0]
        #time_steps = np.arange(0.0, 40.0, dt)
        time_steps = np.linspace(0, self.end, self.n_step)
        self.time_steps = time_steps

        x_train = odeint(f, state0, time_steps)
        return x_train

    def __len__(self):
        #return self.n_step//self.n_step_per_batch
        return (self.n_step-25)//25

    def __getitem__(self, idx):
        #i = idx*self.n_step_per_batch
        i = idx*25
        d=  self.x_train[i:i+self.n_step_per_batch]
        b=  self.basis[i:i+self.n_step_per_batch]
        return i, d, b
#N,T


ds = LorenzDataset(n_step=T,n_step_per_batch=n_step_per_batch)#.generate()
train_loader =DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True) 


target_u = torch.ones(10, dtype=dtype)
#target_u = target_u.permute(1,0)
if cuda:
    target_u = target_u.cuda()
#print('data shape', target_u.shape)

    
class Model(nn.Module):
    def __init__(self, bs, n_step,n_step_per_batch, n_basis, device=None, **kwargs):
        super().__init__()

        self.n_step = n_step #+ 1
        self.order = 2
        # state dimension
        self.bs = bs
        #self.n_coeff = self.n_step * (self.order + 1)
        self.device = device
        self.n_iv=1
        #self.nx = 43
        self.n_ind_dim = 3
        self.n_step_per_batch = n_step_per_batch


        self.n_basis = ds.n_basis

        self.init_xi = torch.tensor(np.random.random((1, self.n_basis, self.n_ind_dim))).type_as(target_u)


        self.mask = torch.ones_like(self.init_xi).type_as(target_u)

        self.step_size = 0.001
        self.steps = self.step_size*torch.ones(self.bs, self.n_ind_dim, self.n_step_per_batch-1)
        self.xi = nn.Parameter(self.init_xi.clone())

        init_coeffs = torch.rand(1, self.n_ind_dim, 1, 2).type_as(target_u)
        self.init_coeffs = nn.Parameter(init_coeffs)
        
        #self.l0_train = ODEForwardINDLayer(bs=bs, order=self.order, step_size=self.step_size, n_ind_dim=self.n_ind_dim, n_step=self.n_step_per_batch, n_iv=self.n_iv, **kwargs)
        self.ode = ODEINDLayer(bs=bs, order=self.order, n_ind_dim=self.n_ind_dim, n_step=self.n_step_per_batch, n_iv=self.n_iv, n_iv_steps=1, **kwargs)

        self.z = torch.zeros(1, self.n_ind_dim, 1,1).type_as(target_u)
        self.o = torch.ones(1, self.n_ind_dim, 1,1).type_as(target_u)

        self.net = nn.Sequential(
            nn.Linear(self.n_ind_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.n_step_per_batch*self.n_ind_dim)
        )
    
    def reset_params(self):
        #self.xi = nn.Parameter(self.init_xi)
        #self.xi.data = self.init_xi.clone()
        self.xi.data = torch.rand_like(self.init_xi)

    def update_mask(self, mask):
        self.mask = self.mask*mask
        #self.mask = mask

    #def set_basis(self, u):
    #    self.basis_tensor,_ = basis.create_library(u, polynomial_order=5, use_trig=False)
    #    self.n_basis = self.basis_tensor.shape[1]

    def forward(self, index, net_iv):
        # apply mask
        xi = self.mask*self.xi
        xi = xi.repeat(self.bs, 1,1)


        var = self.net(net_iv[:,0,:])
        var = var.reshape(self.bs, self.n_step_per_batch, self.n_ind_dim)

        var_basis,_ = B.create_library_tensor_batched(var, polynomial_order=2, use_trig=False, constant=False)

        rhs = var_basis@xi
        rhs = rhs.permute(0,2,1)


        coeffs = torch.cat([self.z,self.o,self.z], dim=-1)
        coeffs = coeffs.repeat(self.bs,1,self.n_step_per_batch,1)

        init_iv = var[:,0]

        self.steps = self.steps.type_as(net_iv)

        x0,x1,x2,eps,steps = self.ode(coeffs, rhs, init_iv, self.steps)
        x0 = x0.permute(0,2,1)

        return x0, steps, eps, var


print('target shape', target_u.shape)
tu = target_u


model = Model(bs=batch_size,n_step=T, n_step_per_batch=n_step_per_batch, n_basis=ds.n_basis, device='cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if DBL:
    model = model.double()
if cuda:
    model=model.cuda()


def print_eq():
    repr_dict = B.basis_repr(model.xi*model.mask, ds.basis_vars)
    for k,v in repr_dict.items():
        L.info(f'{k} = {v}')

def train():
    model.reset_params()

    print_eq()
    optimize()

    for step in range(100):
        #threshold
        params = model.xi
        mask = (params.abs() > threshold).float()
        #ipdb.set_trace()

        L.info(model.xi)
        L.info(model.xi*model.mask)
        L.info(model.mask)
        L.info(model.init_coeffs)
        L.info(model.mask*mask)

        print_eq()
        #set mask

        model.update_mask(mask)
        #optimize


        #print('xxxxxxxxx restarting ', step)
        #print('bef', model.xi, flush=True)
        model.reset_params()
        #print('aft', model.xi, model.init_xi, flush=True)
        optimize()

def train_l1():
    lc = 0.1
    model.reset_params()
    optimize(lc=lc)

    for step in range(9):
        model.reset_params()
        #print('aft', model.xi, model.init_xi, flush=True)
        optimize(lc=2*lc)


def optimize(lc=0):
    for epoch in range(400):
        for i, (index, batch, basis) in enumerate(train_loader):
            batch = batch.type_as(target_u)
            basis = basis.type_as(target_u)

            #print('in', batch.shape, basis.shape)
            optimizer.zero_grad()
            #x0, steps, eps = model(index, batch[:,0,:])
            x0, steps, eps, var = model(index, batch)
            #x0 = x0#.squeeze()
            #ipdb.set_trace()
            loss = (x0- batch).pow(2).mean()
            #loss = (x0- batch).pow(2).sum(dim=[1,2]).mean()
            #loss = (x0- var).pow(2).mean()
            #loss = loss +  (var- batch).pow(2).sum(dim=[1,2]).mean()
            loss = loss +  (var- batch).pow(2).mean()
            #loss = loss + eps.pow(2).mean()

            #loss_l1 = model.xi.abs().sum()
            #loss = loss + lc * loss_l1
            #torch.linalg.lstsq(model.basis, target_u)
            
            loss.backward()
            optimizer.step()

            #print('loss ', loss.item(), model.steps)
        #print(f'id {run_id}. loss ', epoch, loss.item(), steps.squeeze()[0,0].detach().cpu())

        meps = eps.max().item()
        #print(f'run {run_id}. loss ', epoch,  loss.item(), steps.squeeze()[0,0].detach().cpu(), f'eps {meps} ')
        #print(f'run {run_id}. loss ', epoch,  loss.item(), f'eps {meps} ')
        L.info(f'run {run_id} loss {epoch},  {loss.item()} eps {meps} ')
        #print(model.xi)
        #print(model.init_xi)
    #print('out', x0[1:5])

    #print(model.xi)
    #print(model.xi*model.mask)
    #print('coeffs', model.init_coeffs)

def simulate(xi):
    def f(state, t):
        x, y, z = state
        #return sigma * (y - x), x * (rho - z) - y, x * y - beta * z
        #return sigma * (y - x), x * (rho - z) - y, x * y - beta * z
        input = np.array([x,y,z]).reshape((1,3))
        basis,_ =B.create_library(input, polynomial_order=2, use_trig=False)

        rhs = basis@xi

        dx = -0.0879 + 1.6870*x
        dy = -0.8512 + 1.0273*x -0.5622*z -0.0339*x*y +0.1025*y*y
        dz = 0.9278 + 0.0479*x + 1.2808*z
        #return -0.0879, x * (rho - z) - y, x * y - beta * z
        dx = (dx - 1.6824*x)
        dy = (dy - 1.9404*y)
        dz = (dz - 1.3089*z)

        dx = dx/0.8365
        dy = dy/0.9703
        dz = dz/0.3601

        return dx,dy,dz

    state0 = [1.0, 1.0, 1.0]
    #time_steps = np.arange(0.0, 40.0, dt)
    time_steps = np.linspace(0, 2000*0.01, 2000)

    x_sim = odeint(f, state0, time_steps)
    return x_sim

train()
#train_l1()
print(model.xi)
print(model.xi*model.mask)
print('coeffs', model.init_coeffs)

print_eq()

#ipdb.set_trace()