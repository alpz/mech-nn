import torch
from torch.autograd import Function
import numpy as np
#import osqp
import ipdb

#from sksparse.cholmod import cholesky, cholesky_AAt, analyze, analyze_AAt

import scipy.sparse.linalg as spla
import scipy.linalg as spl

import scipy.sparse as SP
import torch.linalg as TLA

def batch_cholesky(A):
    L = torch.zeros_like(A)

    for i in range(A.shape[-1]):
        for j in range(i+1):
            s = 0.0
            for k in range(j):
                s = s + L[...,i,k] * L[...,j,k]

            L[...,i,j] = torch.sqrt(A[...,i,i] - s) if (i == j) else \
                      (1.0 / L[...,j,j] * (A[...,i,j] - s))
    return L


def solve_kkt2(A, L, g, h, gamma):
    """
        Solve min x'Gx + d'x
            Ax = b

            g := d
            h := -b
            p := x*
            G := gamma*I
    """
    
    At = A.transpose(1,2)
    #print('c b', A.shape,  c.shape, b.shape)

    h = h.unsqueeze(2)
    g = g.unsqueeze(2)
    
    rhs1 = A@g - gamma*h
    y = torch.cholesky_solve(rhs1, L)
    y = y
    
    p = At@y - g
    p = p/gamma
    
    #return -p,y
    return p,y

def solve_kkt(A, L, c, b, gamma):
    At = A.transpose(1,2)

    #c = c.unsqueeze(2)
    #b = b.unsqueeze(2)
    #print("A b, c", A.shape, b.shape, c.shape)
    c1 = A @ c.unsqueeze(2)
    #c2 = FAA.solve_A(c1)
    c2 = torch.cholesky_solve(c1,L)
    #c2 = torch.linalg.lu_solve(L,pv,c1)
    # c2,_ = spla.gmres(AAt, c1)
    c3 = At @ c2
    Cc = 1/gamma * (1 - c3)

    #Fnb = -FAA.solve_A(-b)
    #Fnb = -torch.cholesky_solve(-b,L)
    lb = torch.cholesky_solve(b.unsqueeze(2),L)
    Fnb = -gamma*lb
    #Fnb = -lb
    #Fnb = -lb
    #Fnb = -lb
    #Fnb = -torch.linalg.lu_solve(L,pv, b.unsqueeze(2))
    # Fnb = -spla.gmres(AAt, -b)[0]
    #Enb = At @ FAA.solve_A(-b)
    #Enb = At @ torch.cholesky_solve(-b,L)
    #Enb = At @ torch.cholesky_solve(b.unsqueeze(2),L)
    Enb = At @ lb
    #Enb = At @ torch.linalg.lu_solve(L,pv,b.unsqueeze(2))
    e1 = A @ c.unsqueeze(2)
    #Etc = FAA.solve_A(e1)
    Etc = torch.cholesky_solve(e1,L)
    #Etc = torch.linalg.lu_solve(L,pv, e1)
    #print('etc, fnb, enb', Etc, Fnb, Enb)

    x = Cc + Enb
    y = Etc + Fnb

    return x,y


#def QPFunction(ode, n_step=100, order=2, n_iv=2, gamma=0.1, DEVICE='cuda', coeffs=None, rhs=None):
def QPFunction(ode, n_step=100, order=2, n_iv=2, gamma=1, alpha=1, DEVICE='cuda', double_ret=False):
    #gamma = 0.5

    class QPFunctionFn(Function):
        @staticmethod
        #def forward(ctx, coeffs, rhs, iv_rhs, derivative_A):
        def forward(ctx, eq_A, rhs, iv_rhs, derivative_A):
        #def forward(ctx, coeffs, rhs, iv_rhs):
            #bs = coeffs.shape[0]
            bs = rhs.shape[0]
            #ode.build_ode(coeffs, rhs, iv_rhs, derivative_A)
            ode.build_ode(eq_A, rhs, iv_rhs, derivative_A)
            #ode.build_ode(coeffs, rhs, iv_rhs, None)
            
            At = ode.AG.to_dense()

            #A = ode.A
            ub = ode.ub

            #ipdb.set_trace()

            c = torch.zeros(bs, ode.num_vars).type_as(rhs)
            #print("c ", c.dtype, c.shape, At.shape)
            c[:,0] = alpha
            
            b = c
            c = ub.type_as(rhs)
            A = At.transpose(1,2)
            AAt = A@At 
            #L = torch.linalg.cholesky(AAt,upper=False)
            L,info = torch.linalg.cholesky_ex(AAt,upper=False)

            x,y = solve_kkt2(A,L, c, -b, gamma)
            #x,y = solve_kkt2(A,L, c, b, gamma)
            
            
            x = x.squeeze(2)
            y = y.squeeze(2)

            #print('x,y', x,y)
            ctx.save_for_backward(A, L, x, y)
            #print("x,y ", x,y)
            
            if not double_ret:
                y = y.float()
            return y
        
        @staticmethod
        def backward(ctx, dl_dzhat):
            A,L, _x, _y = ctx.saved_tensors
            n = A.shape[1]
            m = A.shape[2]
            At = A.transpose(1,2)
            
            bs = dl_dzhat.shape[0]
            m = ode.num_constraints

            z = torch.zeros(bs, m).type_as(_x)

            _dx,_dnu = solve_kkt2(A,L, z, -dl_dzhat, gamma)
            
            _dx, _dnu = -_dx,-_dnu

            #take row, col indices
            dx = _dx[:,0:n_step].reshape(bs, n_step,1)
            x = _x[:,0:n_step].reshape(bs, n_step,1)
            
            nu = _y
            
            t_vars = ode.n_system_vars
            num_coeffs = t_vars*n_step*(order+1)

            #remove eps
            dnu = _dnu[:, 1:1+num_coeffs]
            nu = nu[:, 1:1+num_coeffs]
            
            #dA = dx*nu.reshape(bs, 1,num_coeffs)
            #dA = dA + x*dnu.reshape(bs, 1,num_coeffs)
            
            ##print('pre d', dA)
            #                  
            #mask = ode.mask_A
            ##mask = SP.bmat([[mask], [mask]],format='csc')
            
            ##mask = ode.mask_A#.todense()
            #mask = mask.to_dense()
            ##dA = (mask*dA).sum(axis=1)
            #dA = (dA).sum(axis=1)

            #div_rhs = _dx[:, n_step:n_step+2*n_iv].squeeze(2)
            div_rhs = _dx[:, n_step*ode.n_equations:(n_step+n_iv)*ode.n_equations].squeeze(2)
            
            #dA = torch.tensor(dA)#.sum(dim=0)
            #db = _dx[:, :2*n_step] #torch.tensor(-dnu.squeeze())
            db = _dx[:, :n_step*ode.n_equations] #torch.tensor(-dnu.squeeze())
            db = -db.reshape(bs,n_step*ode.n_equations)
            #div_rhs = torch.tensor(div_rhs)
            
            #dA = dA.reshape(bs,n_step,t_vars, order+1)

            if ode.n_iv == 0:
                div_rhs = None
            else:
                div_rhs = -div_rhs.reshape(bs,n_iv*ode.n_equations)
            
            #db = -db.reshape(bs,2,n_step)*mm
            #db = db.sum(dim=1)

            # step gradient
            dD = ode.sparse_grad_derivative_constraint(_dx,_y)
            dD = dD + ode.sparse_grad_derivative_constraint(_x,_dnu)

            # eq grad
            dA = ode.sparse_grad_eq_constraint(_dx,_y)
            dA = dA + ode.sparse_grad_eq_constraint(_x,_dnu)

            if not double_ret:
                dA = dA.float()
                db =db.float()
                div_rhs = div_rhs.float() if div_rhs is not None else None
                dD = dD.float()
            
            return dA, db,div_rhs, dD
            #return dA, db,None, dD
            #return dA, db,div_rhs, None
            #return dA, db,div_rhs

    return QPFunctionFn.apply
