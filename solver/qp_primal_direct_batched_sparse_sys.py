import torch
from torch.autograd import Function
import numpy as np

#from sksparse.cholmod import cholesky, cholesky_AAt, analyze, analyze_AAt

import scipy.sparse.linalg as spla
import scipy.linalg as spl

import scipy.sparse as SP
import torch.linalg as TLA



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
    
    return p,y



def QPFunction(ode, n_step=100, order=2, n_iv=2, gamma=1, alpha=1, DEVICE='cuda', double_ret=True):

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
            #minimize gamma*eps^2 +alpha*eps
            c[:,0] = alpha
            
            b = c
            c = ub.type_as(rhs)
            A = At.transpose(1,2)
            AAt = A@At 
            #L = torch.linalg.cholesky(AAt,upper=False)
            L,info = torch.linalg.cholesky_ex(AAt,upper=False)

            x,y = solve_kkt2(A,L, c, -b, gamma)
            
            
            x = x.squeeze(2)
            y = y.squeeze(2)

            ctx.save_for_backward(A, L, x, y)
            
            if not double_ret:
                y = y.float()
            return y

        @staticmethod
        def backward(ctx, dl_dzhat):
            #A,AAt, _x, _y = ctx.saved_tensors
            A,L, _x, _y = ctx.saved_tensors
            
            bs = dl_dzhat.shape[0]
            m = ode.num_constraints

            z = torch.zeros(bs, m).type_as(_x)

            _dx,_dnu = solve_kkt2(A,L, z, -dl_dzhat, gamma)
            
            _dx, _dnu = -_dx,-_dnu

            db = _dx[:, :ode.num_added_equation_constraints] #torch.tensor(-dnu.squeeze())
            db = -db.squeeze(-1) #.reshape(bs,n_step*ode.n_equations)

            if ode.n_iv == 0:
                div_rhs = None
            else:
                #div_rhs = _dx[:, n_step*ode.n_equations:(n_step+n_iv)*ode.n_equations].squeeze(2)
                div_rhs = _dx[:, ode.num_added_equation_constraints:ode.num_added_equation_constraints + ode.num_added_initial_constraints].squeeze(2)
                div_rhs = -div_rhs#.reshape(bs,n_iv*ode.n_equations)
            

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
        

    return QPFunctionFn.apply
