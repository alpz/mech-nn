import sys
sys.path.append("../../")

import torch.nn as nn
import torch
#from lp_dyn_cent_sys_dim import ODESYSLP as ODELP
#from lp_dyn_cent_sys_sparse import ODESYSLP as ODELP_sys
from lp_sparse_forward_diff import ODESYSLP #as ODELP_sys
from torch.nn.parameter import Parameter
import numpy as np

import torch
from qp_primal_direct_batched_sys import QPFunction
from qp_primal_direct_batched_sparse_sys import QPFunction as QPFunctionSys


DBL=False
#class ODEINDLayer(nn.Module):
#    """set of independent ODEs"""
#    def __init__(self, bs, order, n_ind_dim, n_iv, n_step, device=None):
#        super().__init__()
#        # default step size
#        self.step_size = 0.1
#        #self.end = n_step * self.step_size
#        self.n_step = n_step #int(self.end / self.step_size)
#        self.order = order
#
#        # independent dimension
#        self.n_ind_dim = n_ind_dim
#        self.n_iv = n_iv
#        self.bs = bs
#        self.n_coeff = self.n_step * (self.order + 1)
#        self.device = device
#        dtype = torch.float64 if DBL else torch.float32
#
#        #self.ode = ODELP(bs=bs * self.n_ind_dim, n_dim=1, n_auxiliary=0, n_step=self.n_step, step_size=self.step_size, order=self.order,
#        #                 n_iv=self.n_iv, dtype=dtype, device=self.device)
#
#        self.ode = ODESYSLP(bs=bs, n_dim=self.n_dim, n_equations=n_equations, n_auxiliary=0, n_step=self.n_step, step_size=self.step_size, order=self.order,
#                         n_iv=self.n_iv, dtype=dtype, device=self.device)
#
#        #self.steps = 0.1*torch.ones(1,self.n_ind_dim,self.n_step-1,1)
#        #self.steps = torch.logit(0.1*torch.ones(1,self.n_ind_dim,self.n_step-1,1))
#        self.steps = torch.logit(0.5*torch.ones(self.n_step-1).reshape(1,1,self.n_step-1,1))
#        #self.steps = nn.Parameter(self.steps)
#
#        self.qpf = QPFunction(self.ode, n_step=self.n_step, order=self.order, n_iv=self.n_iv, gamma=0.2)
#
#    def forward(self, coeffs, rhs, iv_rhs, steps):
#        coeffs = coeffs.reshape(self.bs * self.n_ind_dim, self.n_step,1, self.order + 1)
#
#
#        rhs = rhs.reshape(self.bs*self.n_ind_dim, self.n_step)
#        if iv_rhs is not None:
#            iv_rhs = iv_rhs.reshape(self.bs * self.n_ind_dim, self.n_iv)
#
#        #steps = self.steps.repeat(self.bs, 1, 1, 1)
#        #steps = self.steps.repeat(self.bs, self.n_ind_dim, self.n_step-1, 1)
#        #steps = self.steps.repeat(self.bs, self.n_ind_dim, 1, 1)
#        steps = steps.reshape(self.bs*self.n_ind_dim,self.n_step-1,1)
#
#
#        #coeffs = torch.tanh(coeffs)
#        #coeffs = torch.relu(coeffs)
#        #coeffs = torch.gelu(coeffs)
#        #coeffs = torch.sigmoid(coeffs)
#        #rhs = torch.sigmoid(rhs)
#        #rhs = torch.gelu(rhs)
#        #iv_rhs = torch.sigmoid(iv_rhs)
#        #rhs = 0*(rhs)
#        #coeffs = coeffs.clone()
#        #coeffs[:,:,:,0] = 0.
#        #coeffs = torch.tanh(coeffs)
#        #rhs = torch.tanh(rhs)
#        #steps =torch.sigmoid(steps).clip(min=0.001)
#        #steps = self.steps
#
#        derivative_constraints = self.ode.build_derivative_tensor(steps)
#
#        if DBL:
#            coeffs = coeffs.double()
#            rhs = rhs.double()
#            iv_rhs = iv_rhs.double() if iv_rhs is not None else None
#
#        x = self.qpf(coeffs, rhs, iv_rhs, derivative_constraints)
#
#        eps = x[:,0]
#
#        #shape: batch, step, vars (== 1), order
#        u = self.ode.get_solution_reshaped(x)
#
#        u = u.reshape(self.bs, self.n_ind_dim, self.n_step, self.order+1)
#        #shape: batch, step, ind_vars, order
#        u = u.permute(0,2,1,3)
#
#        u0 = u[:,:,:,0]
#        u1 = u[:,:,:,1]
#        #u2 = u[:,:,:,2]
#        
#        return u0, u1, eps, steps

class ODEINDLayer(nn.Module):
    def __init__(self, bs, order, n_ind_dim, n_iv, n_step, n_iv_steps, device=None):
        super().__init__()
        # default step size
        self.step_size = 0.1
        #self.end = n_step * self.step_size
        self.n_step = n_step #int(self.end / self.step_size)
        self.order = order

        self.n_ind_dim = n_ind_dim
        self.n_dim = 1 #n_dim
        self.n_equations =1 # n_equations
        self.n_iv = n_iv
        self.n_iv_steps = n_iv_steps
        self.bs = bs
        self.n_coeff = self.n_step * (self.order + 1)
        self.device = device
        dtype = torch.float64 if DBL else torch.float32

        self.ode = ODESYSLP(bs=bs*self.n_ind_dim, n_dim=self.n_dim, n_equations=self.n_equations, n_auxiliary=0, n_step=self.n_step, step_size=self.step_size, order=self.order,
                         n_iv=self.n_iv, n_iv_steps=self.n_iv_steps, dtype=dtype, device=self.device)

        #self.steps = torch.logit(0.5*torch.ones(1,self.n_step-1,self.n_dim))
        #self.steps = (0.1*torch.ones(1,self.n_step-1,self.n_dim))
        #self.steps = nn.Parameter(self.steps)

        self.qpf = QPFunctionSys(self.ode, n_step=self.n_step, order=self.order, n_iv=self.n_iv, gamma=0.5, alpha=0.1, double_ret=False)

    def forward(self, coeffs, rhs, iv_rhs, steps):
        #coeffs = coeffs.reshape(self.bs , self.n_dim, self.n_step,self.n_dim, self.order + 1)
        #coeffs = coeffs.reshape(self.bs , self.n_equations, self.n_step,self.n_dim, self.order + 1)
        coeffs = coeffs.reshape(self.bs*self.n_ind_dim, self.n_step,self.n_dim, self.order + 1)
        #coeffs = coeffs.reshape(self.bs , self.n_dim, 1,self.n_dim, self.order + 1).repeat(1,1,self.n_step,1,1)


        #rhs = rhs.reshape(self.bs, self.n_step* self.n_dim)
        #rhs = rhs.reshape(self.bs, self.n_step* self.n_equations)
        rhs = rhs.reshape(self.bs*self.n_ind_dim, self.n_step)
        if iv_rhs is not None:
            #iv_rhs = iv_rhs.reshape(self.bs*self.n_ind_dim, self.n_iv)
            iv_rhs = iv_rhs.reshape(self.bs*self.n_ind_dim, self.n_iv_steps*self.n_iv)

        #steps = self.steps.repeat(self.bs, 1, 1, 1).type_as(rhs)
        steps = steps.reshape(self.bs*self.n_ind_dim,self.n_step-1,1)


        #coeffs = torch.sigmoid(coeffs)
        #rhs = torch.sigmoid(rhs)
        #coeffs = torch.tanh(coeffs)
        #rhs = torch.tanh(rhs)
        #iv_rhs = torch.tanh(iv_rhs)
        #steps =torch.sigmoid(steps).clip(min=0.001)
        #steps =(steps).clip(min=0.001)
        #steps = self.steps



        if DBL:
            coeffs = coeffs.double()
            rhs = rhs.double()
            iv_rhs = iv_rhs.double() if iv_rhs is not None else None
            steps = steps.double()

        derivative_constraints = self.ode.build_derivative_tensor(steps)
        eq_constraints = self.ode.build_equation_tensor(coeffs)

        #x = self.qpf(coeffs, rhs, iv_rhs, derivative_constraints)
        x = self.qpf(eq_constraints, rhs, iv_rhs, derivative_constraints)

        eps = x[:,0]

        #shape: batch, step, vars (== 1), order
        u = self.ode.get_solution_reshaped(x)

        u = u.reshape(self.bs, self.n_ind_dim, self.n_step, self.order+1)
        #shape: batch, step, vars, order
        #u = u.permute(0,2,1,3)

        u0 = u[:,:,:,0]
        u1 = u[:,:,:,1]
        u2 = u[:,:,:,2]
        
        return u0, u1, u2, eps, steps

class ODESYSLayer(nn.Module):
    def __init__(self, bs, order, n_ind_dim, n_dim, n_equations, n_iv, n_iv_steps, n_step, device=None):
        super().__init__()
        # default step size
        self.step_size = 0.1
        #self.end = n_step * self.step_size
        self.n_step = n_step #int(self.end / self.step_size)
        self.order = order

        self.n_dim = n_dim
        self.n_ind_dim = n_ind_dim
        self.n_equations = n_equations
        self.n_iv = n_iv
        self.n_iv_steps = n_iv_steps

        self.bs = bs
        self.n_coeff = self.n_step * (self.order + 1)
        self.device = device
        dtype = torch.float64 if DBL else torch.float32

        self.ode = ODESYSLP(bs=bs*self.n_ind_dim, n_dim=self.n_dim, n_equations=n_equations, n_auxiliary=0, n_step=self.n_step, step_size=self.step_size, order=self.order,
                         n_iv=self.n_iv, n_iv_steps=self.n_iv_steps, dtype=dtype, device=self.device)

        #self.steps = torch.logit(0.5*torch.ones(1,self.n_step-1,self.n_dim))
        #self.steps = (0.1*torch.ones(1,self.n_step-1,self.n_dim))
        #self.steps = nn.Parameter(self.steps)

        self.qpf = QPFunctionSys(self.ode, n_step=self.n_step, order=self.order, n_iv=self.n_iv, gamma=0.5, alpha=0.1)

    def forward(self, coeffs, rhs, iv_rhs, steps):
        #coeffs = coeffs.reshape(self.bs , self.n_dim, self.n_step,self.n_dim, self.order + 1)
        coeffs = coeffs.reshape(self.bs*self.n_ind_dim, self.n_equations, self.n_step,self.n_dim, self.order + 1)
        #coeffs = coeffs.reshape(self.bs , self.n_dim, 1,self.n_dim, self.order + 1).repeat(1,1,self.n_step,1,1)


        #rhs = rhs.reshape(self.bs, self.n_step* self.n_dim)
        #n_equation, n_step
        rhs = rhs.reshape(self.bs*self.n_ind_dim, self.n_equations*self.n_step)

        #iv_steps, n_dim, n_iv
        iv_rhs = iv_rhs.reshape(self.bs*self.n_ind_dim, self.n_iv_steps*self.n_dim*self.n_iv)

        #steps = self.steps.repeat(self.bs, 1, 1, 1).type_as(rhs)
        steps = steps.reshape(self.bs*self.n_ind_dim,self.n_step-1,self.n_dim)


        #coeffs = torch.sigmoid(coeffs)
        #rhs = torch.sigmoid(rhs)
        #coeffs = torch.tanh(coeffs)
        #rhs = torch.tanh(rhs)
        #iv_rhs = torch.tanh(iv_rhs)
        #steps =torch.sigmoid(steps).clip(min=0.001)
        #steps =(steps).clip(min=0.001)
        #steps = self.steps

        if DBL:
            coeffs = coeffs.double()
            rhs = rhs.double()
            iv_rhs = iv_rhs.double() if iv_rhs is not None else None
            steps = steps.double()

        derivative_constraints = self.ode.build_derivative_tensor(steps)
        eq_constraints = self.ode.build_equation_tensor(coeffs)

        #x = self.qpf(coeffs, rhs, iv_rhs, derivative_constraints)
        x = self.qpf(eq_constraints, rhs, iv_rhs, derivative_constraints)

        eps = x[:,0]

        #shape: batch, step, vars (== 1), order
        u = self.ode.get_solution_reshaped(x)

        u = u.reshape(self.bs,self.n_ind_dim, self.n_step, self.n_dim, self.order+1)
        #shape: batch, step, vars, order
        #u = u.permute(0,2,1,3)

        u0 = u[:,:,:,:,0]
        u1 = u[:,:,:,:,1]
        u2 = u[:,:,:,:,2]
        
        return u0, u1, u2,eps, steps