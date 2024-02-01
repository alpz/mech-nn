import torch
from torch.autograd import Function
import numpy as np

#from sksparse.cholmod import cholesky, cholesky_AAt, analyze, analyze_AAt

import scipy.sparse.linalg as spla
import scipy.linalg as spl

import scipy.sparse as SP
import torch.linalg as TLA


import solver.cg as cg
from config import ODEConfig as config


def block_mv(A, x):
    """shape x: (b, d), A sparse block"""
    b = x.shape[0]
    x = x.reshape(-1)

    y = torch.mv(A, x)
    y = y.reshape(b, -1)
    return y

def solve_kkt_indirect_cg(A, AAt, g, h, gamma):
    """
        Solve min x'Gx + d'x
            Ax = b

            g := d
            h := -b
            p := x*
            G := gamma*I
    """
    At = A.t() #transpose(1,2)
    rhs1 = block_mv(A,g) - gamma*h
    #y = torch.cholesky_solve(rhs1, L)
    #y = y

    #rhs = cupy.asarray(rhs1.squeeze())#.cpu().numpy()
    #print('starting ')
    #rhs = rhs1.squeeze(-1)
    y,info = cg.cg_block(AAt, rhs1, maxiter=config.cg_max_iter)
    #res = rhs1 - block_mv(AAt,y)
    #print('done ', res.mean().item() )
    
    p = block_mv(At,y) - g
    p = p/gamma
    
    return p,y


def QPFunction(ode, n_iv, order, n_step=10, gamma=1, alpha=1, double_ret=True):


    class QPFunctionFn(Function):
        #csr_rows = None
        #csr_cols = None
        #nnz = None
        #QRSolver = None
        #dim = None
        #perm = None

        @staticmethod
        def forward(ctx, eq_A, rhs, iv_rhs, derivative_A):
        #def forward(ctx, coeffs, rhs, iv_rhs):
            #bs = coeffs.shape[0]
            bs = rhs.shape[0]
            #ode.build_ode(coeffs, rhs, iv_rhs, derivative_A)
            At, ub = ode.fill_block_constraints_torch(eq_A, rhs, iv_rhs, derivative_A)
            #ode.build_ode(coeffs, rhs, iv_rhs, None)
            #dtype = torch.float32
            

            #A = ode.A
            #ub = ode.ub

            #ipdb.set_trace()

            c = torch.zeros(bs, ode.num_vars).type_as(rhs)
            #print("c ", c.dtype, c.shape, At.shape)
            #minimize gamma*eps^2 +alpha*eps
            c[:,0] = alpha
            
            b = c
            c = ub.type_as(rhs)

            #A to csr
            #A  split 
            #value_list = []
            #At = ode.AG#.to_dense()
            #print(At.shape)

            A = At.t()
            #AAt = torch.sparse.mm(At.t(), At)
            AAt = torch.sparse.mm(A, At)
            #AAt_list = []
            #for i in range(bs):
            #i=0
            #AAti = At[i].t()@At[i]
            #AAti = AAti.unsqueeze(0)
            #AAti = [(At[i].t()@At[i]).unsqueeze(0) for i in range(bs)]
            #AAti = torch.cat(AAti, dim=0)
            #AAt_sp = tensor_to_cpsp(AAti)
            #    AAt_list.append(AAt_sp)
            #rows = AAti.crow_indices().detach().cpu().numpy()
            #cols = AAti.col_indices().detach().cpu().numpy()

            #shape = AAti.shape
            #indices = AAti._indices()
            #rows = indices[0].cpu().numpy()
            #cols = indices[1].cpu().numpy()
            #values = AAti._values().cpu().numpy()

            #AAt_sp = SP.coo_matrix((values, (rows,cols)), shape=shape)
            #AAt_sp = tensor_to_sp(AAti)
            #AAt_sp = tensor_to_cpsp(AAti)

            #x,y = solve_kkt_sparse_qr(A,c, -b, gamma, QPFunctionFn.QRSolver, values)
            #x,y = solve_kkt_indirect(A,AAt_sp,c, -b, gamma)
            #x,y = solve_kkt_indirect(A,AAt_sp,c, -b, gamma)
            x,y = solve_kkt_indirect_cg(A,AAt,c, -b, gamma)
            
            
            ctx.save_for_backward(A, AAt, x, y)
            
            if not double_ret:
                y = y.float()
            return y
        
        @staticmethod
        def backward(ctx, dl_dzhat):
            A,AAt, _x, _y = ctx.saved_tensors
            #n = A.shape[1]
            #m = A.shape[2]
            #At = A.transpose(1,2)
            
            bs = dl_dzhat.shape[0]
            m = ode.num_constraints

            z = torch.zeros(bs, m).type_as(_x)

            #AAt_sp = tensor_to_sp(AAti)
            #AAt_sp = tensor_to_cpsp(AAti)
            #_dx,_dnu = solve_kkt2(A,L, z, -dl_dzhat, gamma)
            #_dx,_dnu = solve_kkt_sparse_qr(A,z, -dl_dzhat, gamma, QPFunctionFn.QRSolver, values)
            #_dx,_dnu = solve_kkt_indirect_cp(A,AAt_sp,z, -dl_dzhat, gamma)
            _dx,_dnu = solve_kkt_indirect_cg(A, AAt, z, -dl_dzhat, gamma)
            
            _dx, _dnu = -_dx,-_dnu

            #take row, col indices
            #dx = _dx[:,0:n_step].reshape(bs, n_step,1)
            #x = _x[:,0:n_step].reshape(bs, n_step,1)
            
            #nu = _y
            
            #t_vars = ode.n_system_vars
            #num_coeffs = t_vars*n_step*(order+1)

            #remove eps
            #dnu = _dnu[:, 1:1+num_coeffs]
            #nu = nu[:, 1:1+num_coeffs]
            

            
            #dA = torch.tensor(dA)#.sum(dim=0)
            #db = _dx[:, :2*n_step] #torch.tensor(-dnu.squeeze())
            #db = _dx[:, :n_step*ode.n_equations] #torch.tensor(-dnu.squeeze())
            db = _dx[:, :ode.num_added_equation_constraints] #torch.tensor(-dnu.squeeze())
            db = -db.squeeze(-1) #.reshape(bs,n_step*ode.n_equations)
            #div_rhs = torch.tensor(div_rhs)
            
            #dA = dA.reshape(bs,n_step,t_vars, order+1)

            if ode.n_iv == 0:
                div_rhs = None
            else:
                #div_rhs = _dx[:, n_step*ode.n_equations:(n_step+n_iv)*ode.n_equations].squeeze(2)
                div_rhs = _dx[:, ode.num_added_equation_constraints:ode.num_added_equation_constraints + ode.num_added_initial_constraints]#.squeeze(2)
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
