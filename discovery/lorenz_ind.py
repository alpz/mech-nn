
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.autograd import gradcheck


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
L = logger.setup(log_dir, stdout=False)

DBL=True
dtype = torch.float64 if DBL else torch.float32
STEP = 0.001
cuda=True
T = 40000
n_step_per_batch = 50
batch_size= 20
#weights less than threshold (absolute) are set to 0 after each optimization step.
threshold = 0.1


class LorenzDataset(Dataset):
    def __init__(self, n_step_per_batch=100, n_step=1000):
        self.n_step_per_batch=n_step_per_batch
        self.n_step=n_step
        self.end= n_step*STEP
        x_train = self.generate()

        self.down_sample = 1

        self.x_train = torch.tensor(x_train, dtype=dtype) 
        self.x_train = self.x_train 

        #Create basis for some stats. Actual basis is in the model
        basis,basis_vars =B.create_library(x_train, polynomial_order=2, use_trig=False, constant=True)
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
        time_steps = np.linspace(0, self.end, self.n_step)
        self.time_steps = time_steps

        x_train = odeint(f, state0, time_steps)
        return x_train

    def __len__(self):
        return (self.n_step-self.n_step_per_batch)//self.down_sample

    def __getitem__(self, idx):
        i = idx*self.down_sample
        d=  self.x_train[i:i+self.n_step_per_batch]
        return i, d


ds = LorenzDataset(n_step=T,n_step_per_batch=n_step_per_batch)#.generate()
train_loader =DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True) 

#plot train data
P.plot_lorenz(ds.x_train, os.path.join(log_dir, 'train.pdf'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, bs, n_step,n_step_per_batch, n_basis, device=None, **kwargs):
        super().__init__()

        self.n_step = n_step #+ 1
        self.order = 2
        # state dimension
        self.bs = bs
        self.device = device
        self.n_iv=1
        self.n_ind_dim = 3
        self.n_step_per_batch = n_step_per_batch

        self.n_basis = ds.n_basis

        self.init_xi = torch.randn((1, self.n_basis, self.n_ind_dim), dtype=dtype).to(device)

        self.mask = torch.ones_like(self.init_xi).to(device)

        #Step size is fixed. Make this a parameter for learned step
        self.step_size = 0.001
        self.xi = nn.Parameter(self.init_xi.clone())

        init_coeffs = torch.rand(1, self.n_ind_dim, 1, 2, dtype=dtype)
        self.init_coeffs = nn.Parameter(init_coeffs)
        
        self.ode = ODEINDLayer(bs=bs, order=self.order, n_ind_dim=self.n_ind_dim, n_step=self.n_step_per_batch, 
                                    n_iv=self.n_iv, n_iv_steps=1, cent_diff=True, gamma=0.3, alpha=1, **kwargs)


        self.net = nn.Sequential(
            nn.Linear(self.n_step_per_batch*self.n_ind_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.n_step_per_batch*self.n_ind_dim)
        )
    
    def reset_params(self):
        #reset basis weights to random values
        self.xi.data = torch.randn_like(self.init_xi)

    def update_mask(self, mask):
        self.mask = self.mask*mask

    def forward(self, index, net_iv):
        # apply mask
        xi = self.mask*self.xi
        xi = xi.repeat(self.bs, 1,1)


        var = self.net(net_iv.reshape(self.bs,-1))
        var = var.reshape(self.bs, self.n_step_per_batch, self.n_ind_dim)

        #create basis
        var_basis,_ = B.create_library_tensor_batched(var, polynomial_order=2, use_trig=False, constant=True)

        rhs = var_basis@xi
        rhs = rhs.permute(0,2,1)

        z = torch.zeros(1, self.n_ind_dim, 1,1).type_as(net_iv)
        o = torch.ones(1, self.n_ind_dim, 1,1).type_as(net_iv)

        coeffs = torch.cat([z,o,z], dim=-1)
        coeffs = coeffs.repeat(self.bs,1,self.n_step_per_batch,1)

        init_iv = var[:,0]

        steps = self.step_size*torch.ones(self.bs, self.n_ind_dim, self.n_step_per_batch-1).type_as(net_iv)
        #self.steps = self.steps.type_as(net_iv)

        x0,x1,x2,eps,steps = self.ode(coeffs, rhs, init_iv, steps)
        x0 = x0.permute(0,2,1)

        return x0, steps, eps, var

model = Model(bs=batch_size,n_step=T, n_step_per_batch=n_step_per_batch, n_basis=ds.n_basis, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

if DBL:
    model = model.double()
model=model.to(device)


def print_eq(stdout=False):
    #print learned equation
    repr_dict = B.basis_repr(model.xi*model.mask, ds.basis_vars)
    code = []
    for k,v in repr_dict.items():
        L.info(f'{k} = {v}')
        if stdout:
            print(f'{k} = {v}')
        code.append(f'{v}')
    return code

def simulate(gen_code):
    #simulate learned equation
    def f(state, t):
        x0, x1, x2= state

        dx0 = eval(gen_code[0])
        dx1 = eval(gen_code[1])
        dx2 = eval(gen_code[2])

        return dx0, dx1, dx2
        
    state0 = [1.0, 1.0, 1.0]
    time_steps = np.linspace(0, T*STEP, T)

    x_sim = odeint(f, state0, time_steps)
    return x_sim

def train():
    """Optimize and threshold cycle"""
    model.reset_params()

    max_iter = 10
    for step in range(max_iter):
        print(f'Optimizer iteration {step}/{max_iter}')

        #threshold
        if step > 0:
            params = model.xi
            mask = (params.abs() > threshold).float()

            L.info(model.xi)
            L.info(model.xi*model.mask)
            L.info(model.mask)
            #L.info(model.init_coeffs)
            L.info(model.mask*mask)

        code = print_eq(stdout=True)
        #simulate and plot

        x_sim = simulate(code)
        P.plot_lorenz(x_sim, os.path.join(log_dir, f'sim_{step}.pdf'))

        #set mask
        if step > 0:
            model.update_mask(mask)
            model.reset_params()

        optimize()


def optimize(nepoch=400):
    with tqdm(total=nepoch) as pbar:
        for epoch in range(nepoch):
            pbar.update(1)
            for i, (index, batch_in) in enumerate(train_loader):
                batch_in = batch_in.to(device)

                optimizer.zero_grad()
                x0, steps, eps, var = model(index, batch_in)

                x_loss = (x0- batch_in).pow(2).mean()
                loss = x_loss +  (var- batch_in).pow(2).mean()

                loss.backward()
                optimizer.step()


            meps = eps.max().item()
            L.info(f'run {run_id} epoch {epoch}, loss {loss.item()} max eps {meps} xloss {x_loss} ')
            pbar.set_description(f'run {run_id} epoch {epoch}, loss {loss.item()} max eps {meps} xloss {x_loss} ')


if __name__ == "__main__":
    train()

    print_eq()
