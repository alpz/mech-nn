import numpy as np
import torch
import math
import scipy.sparse as sp
import scipy.optimize as spopt
import torch.nn as nn
from enum import Enum, IntEnum
import ipdb

class VarType(Enum):
    EPS = 1
    Mesh = 10

class ConstraintType(Enum):
    Equation = 1
    Initial = 10
    Derivative = 20

class Const(IntEnum):
    #placeholder
    PH = -100 

class ODESYSLP(nn.Module):
    def __init__(self, bs=1, n_step=3, n_dim=1,  n_iv=2, n_auxiliary=0, n_equations=1, step_size=0.25, order=2, dtype=torch.float32, n_iv_steps=1, step_list = None, device=None):
        super().__init__()
        
        self.n_step = n_step
        self.step_size = step_size
        if step_list is None:
            step_list = step_size*np.ones(n_step-1)
        self.step_list = step_list

        #initial constraint steps starting from step 0
        self.n_iv_steps = n_iv_steps

        self.num_constraints = 0

        #tracks number of added constraints
        self.num_added_constraints = 0
        self.num_added_equation_constraints = 0
        self.num_added_initial_constraints = 0
        self.num_added_derivative_constraints = 0

        # order is diffeq order. n_order is total number of terms: y'', y', y for order 2.
        self.n_order = order+1
        # number of initial values
        self.n_iv = n_iv
        # number of ode variables
        self.n_dim = n_dim
        # number of auxiliary variables per dim for non-linear terms
        self.n_auxiliary = n_auxiliary
        # dimensions plus n_auxliary vars for each dim
        self.n_system_vars = self.n_dim + self.n_dim*self.n_auxiliary
        self.n_equations = n_equations
        #batch size
        self.bs = bs
        self.dtype = dtype
        self.device = device
        #print('lp device', self.device)

        # total number of qp variables
        #self.num_vars = self.n_step*self.n_order+1
        self.num_vars = self.n_system_vars*self.n_step*self.n_order+1
        # Variables except eps. Used for raveling
        #self.multi_index_shape = (self.n_step, self.n_dim*self.n_auxiliary, self.n_order)
        self.multi_index_shape = (self.n_step, self.n_system_vars, self.n_order)

        #### sparse constraint arrays
        # constraint coefficients
        self.value_dict = {ConstraintType.Equation: [], ConstraintType.Initial: [], ConstraintType.Derivative: []}
        # constraint indices
        self.row_dict = {ConstraintType.Equation: [], ConstraintType.Initial: [], ConstraintType.Derivative: []}
        # variable indices
        self.col_dict = {ConstraintType.Equation: [], ConstraintType.Initial: [], ConstraintType.Derivative: []}
        # rhs values
        self.rhs_dict = {ConstraintType.Equation: [], ConstraintType.Initial: [], ConstraintType.Derivative: []}

        # mask values
        self.mask_value_dict = {ConstraintType.Equation: [], ConstraintType.Initial: [], ConstraintType.Derivative: []}

        # build skeleton constraints. filled during training
        self.build_constraints()

    def get_solution_reshaped(self, x):
        """remove eps and reshape solution"""
        x = x[:, 1:]
        x = x.reshape(-1, *self.multi_index_shape)
        return x

    def get_variable_index_from_multiindex(self, index=None, var_type=VarType.Mesh):
        if var_type == VarType.EPS:
            #eps has index 0
            return 0

        # 0 is epsilon, step, grad_index
        offset = 1

        #index = self.get_coefficient_index(mesh_index, grad_index)
        out_index = np.ravel_multi_index(index, self.multi_index_shape, order='C')
        out_index = offset + out_index

        return out_index

    def add_constraint(self, var_list, values, rhs, constraint_type):
        """ var_list: list of multindex tuples or eps enum """

        if constraint_type == ConstraintType.Equation:
            constraint_index = self.num_added_equation_constraints 
        elif constraint_type == ConstraintType.Initial:
            constraint_index = self.num_added_initial_constraints
        elif constraint_type == ConstraintType.Derivative:
            constraint_index = self.num_added_derivative_constraints

        for i,v in enumerate(var_list):
            if v == VarType.EPS:
                var_index = self.get_variable_index_from_multiindex(None, var_type=VarType.EPS)
                #var_index = self.get_variable_index(None, None, var_type=VarType.EPS)
            else:
                var_index = self.get_variable_index_from_multiindex(v, var_type=VarType.Mesh)
                #var_index = self.get_variable_index(v[0], v[1], var_type=VarType.Mesh)

            self.col_dict[constraint_type].append(var_index)
            self.value_dict[constraint_type].append(values[i])
            #self.row_dict[constraint_type].append(self.num_added_constraints)
            self.row_dict[constraint_type].append(constraint_index)
        
        self.rhs_dict[constraint_type].append(rhs)

        self.num_added_constraints = self.num_added_constraints+1
        if constraint_type == ConstraintType.Equation:
            self.num_added_equation_constraints += 1
        elif constraint_type == ConstraintType.Initial:
            self.num_added_initial_constraints += 1
        elif constraint_type == ConstraintType.Derivative:
            self.num_added_derivative_constraints += 1

    def add_mask(self, mask_values, constraint_type):
        """ mask values for constraint """
        self.mask_value_dict[constraint_type].extend(mask_values)
        

    def build_initial_constraints(self):
        #equation coefficients over grid

        for step in range(self.n_iv_steps):
            for dim in range(self.n_dim):
                for i in range(self.n_iv):
                    #self.add_constraint(var_list = [(0,dim,i)], values=[1], rhs=Const.PH, constraint_type=ConstraintType.Initial)
                    self.add_constraint(var_list = [(step,dim,i)], values=[1], rhs=Const.PH, constraint_type=ConstraintType.Initial)
                    #self.add_constraint(var_list = [(1,dim,i)], values=[1], rhs=Const.PH, constraint_type=ConstraintType.Initial)

    def build_equation_constraints(self):
        #+ve
        # one equation for each dimension
        for e in range(self.n_equations):
            for step in range(self.n_step):
                var_list = []
                val_list = []
                for dim in range(self.n_system_vars):
                    for order in range(self.n_order):
                        var_list.append((step,dim,order))
                        val_list.append(Const.PH)

                self.add_constraint(var_list = var_list, values=val_list, rhs=Const.PH, constraint_type=ConstraintType.Equation)

    def build_derivative_constraints(self):
        
        #central constraints
        #def central_c(var_order):
        def central_c():
            #central difference for derivatives
            for step in range(1, self.n_step-1):
                for dim in range(self.n_system_vars):
                    for var_order in range(1, self.n_order):
                        h = self.step_size
                        self.add_constraint(var_list = [ VarType.EPS, (step-1, dim, var_order-1), (step, dim, var_order-1), (step+1,dim, var_order-1), (step,dim, var_order)], 
                                        #values= [ 1,            -0.5/h,                0,                    0.5/h,                -1], 
                                        values= [ -1,            -0.5/h,                0,                    0.5/h,                -1], 
                                        rhs=0, constraint_type=ConstraintType.Derivative)
        
        #forward constraints
        def forward_c(sign=1):
            for step in range(self.n_step-1):
                for dim in range(self.n_system_vars):
                    #TODO handle corners for derivatives
                    #for i in range(1):
                    for i in range(self.n_order-1):
                        var_list = []
                        val_list = []

                        #epsilon
                        var_list.append(VarType.EPS)
                        val_list.append(1)

                        for j in range(i,self.n_order):
                            #h = self.step_size**(j)
                            h = self.step_list[step]**(j)
                            d = math.factorial(j-i)
                            h = h/d

                            var_list.append((step,dim, j))
                            val_list.append(sign*h)

                        #h = self.step_size**i
                        h = self.step_list[step]**i

                        var_list.append((step+1,dim, i))
                        val_list.append(-sign*h)

                        self.add_constraint(var_list=var_list, values=val_list, rhs=0, constraint_type=ConstraintType.Derivative)


        #backward constraints
        def backward_c(sign=1):
            #for step in reversed(range(1,self.n_step)):
            for step in range(1,self.n_step):
                for dim in range(self.n_system_vars):
                    for i in range(self.n_order-1):
                        var_list = []
                        val_list = []
                    #for i in range(1):
                        #epsilon
                        var_list.append(VarType.EPS)
                        val_list.append(1)

                        for j in range(i,self.n_order):
                            #h = (-self.step_size)**(j)
                            h = (-self.step_list[step-1])**(j)
                            d = math.factorial(j-i)
                            h = h/d

                            var_list.append((step,dim, j))
                            val_list.append(sign*h)

                        #h = (-self.step_size)**i
                        h = (-self.step_list[step-1])**i

                        var_list.append((step-1,dim, i))
                        val_list.append(-sign*h)

                        self.add_constraint(var_list=var_list, values=val_list, rhs=0, constraint_type=ConstraintType.Derivative)


        forward_c(sign=1)
        forward_c(sign=-1)

        #print('adding central')
        #for i in range(1, self.n_order):
            #central_c(var_order=i)
        #central_c()

        backward_c(sign=1)
        backward_c(sign=-1)


    def build_constraints(self):
        
        self.build_equation_constraints()
        self.build_derivative_constraints()
        self.build_initial_constraints()

        eq_A = torch.sparse_coo_tensor([self.row_dict[ConstraintType.Equation],self.col_dict[ConstraintType.Equation]],
                                       self.value_dict[ConstraintType.Equation], 
                                       size=(self.num_added_equation_constraints, self.num_vars), 
                                       #size=(self.num_added_constraints, self.num_vars), 
                                       dtype=self.dtype, device=self.device)
        
        #mask

        eq_rows = np.array(self.row_dict[ConstraintType.Equation])
        eq_columns = np.array(self.col_dict[ConstraintType.Equation])

        #ones = np.ones(eq_rows.shape[0]//2)
        ones = np.ones(eq_rows.shape[0])
        #mask_values = np.concatenate([ones, -1*ones])
        mask_values = ones #np.concatenate([ones, ones])
        mask_A = torch.sparse_coo_tensor([eq_rows,eq_columns-1],mask_values, 
                                         size=(self.num_added_equation_constraints, self.num_vars-1), 
                                         dtype=self.dtype, device=self.device)


        if self.n_iv > 0:
            initial_A = torch.sparse_coo_tensor([self.row_dict[ConstraintType.Initial],self.col_dict[ConstraintType.Initial]],
                                       self.value_dict[ConstraintType.Initial], 
                                       size=(self.num_added_initial_constraints, self.num_vars), 
                                       dtype=self.dtype, device=self.device)
        else:
            initial_A =None


        #print('d vals', len(self.row_dict[ConstraintType.Derivative]))
        derivative_A = torch.sparse_coo_tensor([self.row_dict[ConstraintType.Derivative],self.col_dict[ConstraintType.Derivative]],
                                       self.value_dict[ConstraintType.Derivative], 
                                       #built_values.squeeze(),
                                       size=(self.num_added_derivative_constraints, self.num_vars), 
                                       #size=(self.num_added_constraints, self.num_vars), 
                                       dtype=self.dtype, device=self.device)


        derivative_rhs = self.rhs_dict[ConstraintType.Derivative]
        derivative_rhs = torch.tensor(derivative_rhs, dtype=self.dtype, device=self.device)
        #self.derivative_lb = -dub


        self.set_row_col_sorted_indices()


        #Add batch dim
        #(b, r1, c)
        eq_A = eq_A.unsqueeze(0)
        eq_A = torch.cat([eq_A]*self.bs, dim=0)
        eq_A = eq_A.coalesce()
        
        mask_A = mask_A.unsqueeze(0)
        mask_A = torch.cat([mask_A]*self.bs, dim=0)
        mask_A = mask_A.coalesce()


        #(b, r2, c)
        if initial_A is not None:
            initial_A = initial_A.unsqueeze(0)
            initial_A = torch.cat([initial_A]*self.bs, dim=0)

        #(b, r3, c)
        derivative_A = derivative_A.unsqueeze(0)
        derivative_A = torch.cat([derivative_A]*self.bs, dim=0)

        derivative_rhs = derivative_rhs.unsqueeze(0).repeat(self.bs,1)

        self.derivative_rhs = derivative_rhs
        self.eq_A = eq_A
        #self.register_buffer("mask_A", mask_A)
        self.initial_A =  initial_A
        self.derivative_A =  derivative_A
    
    def get_row_col_sorted_indices(self, row, col, exclude_eps=True):
        """ Compute indices sorted by row and column and repeats. Useful for sparse outer product when computing constraint derivatives"""
        indices = np.stack([row, col], axis=0)

        row_sorted = indices[:, indices[0,:].argsort()]
        column_sorted = indices[:, indices[1,:].argsort()]

        _, row_counts = np.unique(row_sorted[0], return_counts=True)
        _, column_counts = np.unique(column_sorted[1], return_counts=True)

        row_count = row.shape[0]
        #add batch dimension
        batch_dim = torch.arange(self.bs).repeat_interleave(repeats=row_count).unsqueeze(0)

        row_sorted = torch.tensor(row_sorted)
        column_sorted = torch.tensor(column_sorted)

        row_sorted = row_sorted.repeat(1, self.bs)
        column_sorted = column_sorted.repeat(1, self.bs)

        row_sorted = torch.cat([batch_dim, row_sorted], dim=0)
        column_sorted = torch.cat([batch_dim, column_sorted], dim=0)

        #ipdb.set_trace()
        
        return row_sorted, column_sorted, row_counts, column_counts

    def set_row_col_sorted_indices(self):
        ############## derivative indices sorted and counted
        derivative_rows = np.array(self.row_dict[ConstraintType.Derivative])
        derivative_columns = np.array(self.col_dict[ConstraintType.Derivative])
        row_sorted, column_sorted, row_counts, column_counts = self.get_row_col_sorted_indices(derivative_rows, derivative_columns)
        

        self.derivative_row_sorted = torch.tensor(row_sorted)
        self.derivative_column_sorted = torch.tensor(column_sorted)
        self.derivative_row_counts = torch.tensor(row_counts)
        self.derivative_column_counts = torch.tensor(column_counts)
        ##############


        ###############equation indices sorted and counted
        eq_rows = np.array(self.row_dict[ConstraintType.Equation])
        eq_columns = np.array(self.col_dict[ConstraintType.Equation])
        row_sorted, column_sorted, row_counts, column_counts = self.get_row_col_sorted_indices(eq_rows, eq_columns)

        self.eq_row_sorted = torch.tensor(row_sorted)
        self.eq_column_sorted = torch.tensor(column_sorted)
        self.eq_row_counts = torch.tensor(row_counts)
        self.eq_column_counts = torch.tensor(column_counts)
        #################

    #build values for derivative constraints
    def build_central_values(self, steps):
        #steps shape b,  n_step-1, n_system_vars,
        b = steps.shape[0]
        csteps = steps[:, 1:, :]#.unsqueeze(-1)
        ones = torch.ones_like(csteps)
        zeros = torch.zeros_like(csteps)
        psteps = steps[:,:-1, :]#.unsqueeze(-1)

        sum_inv = 1/(csteps + psteps)

        #shape: b, n_steps-1, 4
        #values = torch.cat([ones, -sum_inv, zeros, sum_inv ], dim=-1)
        values = torch.stack([ones, -sum_inv, zeros, sum_inv, -ones ], dim=-1)
        #repeat n_order-1 times
        values = values.unsqueeze(-2).repeat(1,1,1,self.n_order-1,1)

        #flatten
        #shape, b, n_step-1, n_system_vars, n_order-1, 5
        values = values.reshape(b,-1)

        return values

    def _build_forward_values(self, steps,sign=+1):
        b = steps.shape[0]
        #ones = torch.ones_like(steps)
        order_list = []
        for i in range(self.n_order-1):
            order_list.append(torch.ones_like(steps))
            for j in range(i,self.n_order):
                h = steps**(j)
                d = math.factorial(j-i)
                h = h/d
                order_list.append(sign*h)

            #order_list.append(-sign*steps)
            h = (steps)**i
            #order_list.append(-sign*ones)
            order_list.append(-sign*h)

        values = torch.stack(order_list, dim=-1)
        #print(values.shape)

        values = values.reshape(b,-1)
        return values

    
    def build_forward_values(self, steps):
        values_p = self._build_forward_values(steps,sign=+1)
        values_n = self._build_forward_values(steps,sign=-1)

        values = torch.cat([values_p, values_n], dim=-1)

        return values

    def _build_backward_values(self, steps, sign=+1):
        b = steps.shape[0]

        #no reversing
        #ones = torch.ones_like(steps)
        order_list = []
        for i in range(self.n_order-1):
            order_list.append(torch.ones_like(steps))
            for j in range(i,self.n_order):
                #h = (-self.step_size)**(j)
                h = (-steps)**(j)
                d = math.factorial(j-i)
                h = h/d
                #var_list.append((step,dim, j))
                #val_list.append(sign*h)
                order_list.append(sign*h)

            #h = (-self.step_size)#**i
            #h = (-self.step_size)**i
            h = (-steps)**i

            #order_list.append(-sign*(-steps))
            #order_list.append(-sign*ones)
            order_list.append(-sign*h)

        #var_list.append((step-1,dim, i))
        #val_list.append(-sign*h)
        values = torch.stack(order_list, dim=-1)

        # b, n_steps-1, n_system_vars, n_order+2
        values = values.reshape(b,-1)

        return values

    def build_backward_values(self, steps):
        values_p = self._build_backward_values(steps,sign=+1)
        values_n = self._build_backward_values(steps,sign=-1)

        values = torch.cat([values_p, values_n], dim=-1)

        return values

    def build_derivative_values(self, steps):
        #steps = [self.step_size]*(self.n_step-1)
        #steps = torch.tensor(steps).reshape(1,self.n_step-1,1)

        #true_value = torch.tensor(self.value_dict[ConstraintType.Derivative])

        #cv = self.build_central_values(steps)
        fv = self.build_forward_values(steps)
        bv = self.build_backward_values(steps)

        #built_values = torch.cat([fv,cv,bv], dim=-1)
        built_values = torch.cat([fv,bv], dim=-1)

        return built_values


    def build_equation_tensor(self, eq_values):
        #eq_values = self.build_equation_values(steps).reshape(-1)
        #shape batch, n_eq, n_step, n_vars, order+1
        eq_values = eq_values.reshape(-1)

        eq_indices = self.eq_A._indices()
        G = torch.sparse_coo_tensor(eq_indices, eq_values, dtype=self.dtype, device=eq_values.device)

        return G
    
    def build_derivative_tensor(self, steps):
        derivative_values = self.build_derivative_values(steps).reshape(-1)

        #print('built', len(derivative_values))
        derivative_indices = self.derivative_A._indices()
        G = torch.sparse_coo_tensor(derivative_indices, derivative_values, dtype=self.dtype, device=steps.device)

        #return G, derivative_values
        return G#, derivative_values

    #def fill_constraints_torch(self, eq_values, eq_rhs, iv_rhs, derivative_A):
    def fill_constraints_torch(self, eq_A, eq_rhs, iv_rhs, derivative_A):
        bs = eq_rhs.shape[0]

        # (b, *)
        self.constraint_rhs = eq_rhs
        self.initial_rhs = iv_rhs

        self.derivative_rhs = self.derivative_rhs.type_as(eq_rhs)

        #ipdb.set_trace()


        if derivative_A is None:
            G = self.derivative_A
        else:
            G = derivative_A
            #print(G.to_dense())
        #G = G.type_as(constraint_A)

        if self.initial_A is not None:
            initial_A = self.initial_A.type_as(G)

        #print(self.constraint_A.shape, initial_A.shape, G.shape, flush=True)
        if self.initial_A is not None:
            self.AG = torch.cat([eq_A, initial_A, G], dim=1)
        else:
            self.AG = torch.cat([eq_A, G], dim=1)
        #self.AG = torch.cat([constraint_A, G], dim=1)
        #print('AG ', self.AG.shape, flush=True)

        self.num_constraints = self.AG.shape[1]
        #self.ub = torch.cat([self.constraint_rhs, self.boundary_rhs, self.derivative_ub], axis=1)

        if self.initial_A is not None:
            self.ub = torch.cat([self.constraint_rhs, self.initial_rhs, self.derivative_rhs], axis=1)
        else:
            self.ub = torch.cat([self.constraint_rhs, self.derivative_rhs], axis=1)
        #print('ub ', self.ub.shape, flush=True)

    def build_ode(self, coeffs, rhs, iv_rhs, derivative_A):
        order = self.n_order
        
        #coeffs = torch.cat([coeffs, -coeffs], dim=1)
        #rhs = torch.cat([rhs, -rhs], dim=1)
        #iv_rhs = torch.cat([iv_rhs, -iv_rhs], dim=1)
        
        self.fill_constraints_torch(coeffs, rhs, iv_rhs, derivative_A)
    
    def sparse_grad_derivative_constraint(self, x, y):
        """ sparse x y' for derivative constraint"""
        #dx = _dx[:,0:n_step].reshape(bs, n_step,1)
        #dA = dx*nu.reshape(bs, 1,num_coeffs)
        #correct x, y shapes

        b = x.shape[0]
        #copy x across columns. copy y across rows
        x = x[:, self.num_added_equation_constraints+self.num_added_initial_constraints: self.num_added_equation_constraints+self.num_added_initial_constraints+self.num_added_derivative_constraints]
        y = y[:, :self.num_vars]

        #x = x.reshape(b, -1, 1)
        #y = y.reshape(b, 1, -1)

        #dA = x*y.reshape(b, 1,-1)
        #return dA

        x = x.reshape(b,-1)
        y = y.reshape(b,-1)

        self.derivative_row_counts = self.derivative_row_counts.to(x.device)
        self.derivative_column_counts = self.derivative_column_counts.to(x.device)

        x_repeat = torch.repeat_interleave(x, self.derivative_row_counts, dim=-1)
        y_repeat = torch.repeat_interleave(y, self.derivative_column_counts, dim=-1)

        x_repeat = x_repeat.reshape(-1)
        y_repeat = y_repeat.reshape(-1)

        X = torch.sparse_coo_tensor(self.derivative_row_sorted, x_repeat, 
                                       #size=(self.num_added_derivative_constraints, self.num_vars), 
                                       dtype=self.dtype, device=x.device)

        Y = torch.sparse_coo_tensor(self.derivative_column_sorted, y_repeat, 
                                       #size=(self.num_added_derivative_constraints, self.num_vars), 
                                       dtype=self.dtype, device=x.device)

        #ipdb.set_trace()

        dD = X*Y

        return dD

    def sparse_grad_eq_constraint(self, x, y):
        """ sparse x y' for eq constraint"""
        #dx = _dx[:,0:n_step].reshape(bs, n_step,1)
        #dA = dx*nu.reshape(bs, 1,num_coeffs)
        #correct x, y shapes

        b = x.shape[0]
        #copy x across columns. copy y across rows
        x = x[:, 0:self.num_added_equation_constraints]
        y = y[:, 1:self.num_vars]

        #x = x.reshape(b, -1, 1)
        #y = y.reshape(b, 1, -1)

        #dA = x*y.reshape(b, 1,-1)
        #return dA

        x = x.reshape(b,-1)
        y = y.reshape(b,-1)

        self.eq_row_counts = self.eq_row_counts.to(x.device)
        self.eq_column_counts = self.eq_column_counts.to(x.device)

        x_repeat = torch.repeat_interleave(x, self.eq_row_counts, dim=-1)
        y_repeat = torch.repeat_interleave(y, self.eq_column_counts, dim=-1)

        x_repeat = x_repeat.reshape(-1)
        y_repeat = y_repeat.reshape(-1)

        X = torch.sparse_coo_tensor(self.eq_row_sorted, x_repeat, 
                                       #size=(self.num_added_derivative_constraints, self.num_vars), 
                                       dtype=self.dtype, device=x.device)

        Y = torch.sparse_coo_tensor(self.eq_column_sorted, y_repeat, 
                                       #size=(self.num_added_derivative_constraints, self.num_vars), 
                                       dtype=self.dtype, device=x.device)

        #ipdb.set_trace()

        dD = X*Y

        return dD


def test():
        n_step = 10
        dim =1
        #steps = 0.1*torch.ones(1,n_step-1,dim)
        _steps = 0.01+ np.random.random(n_step-1)
        steps = torch.tensor(_steps).reshape(1,n_step-1,1)

        ode = ODESYSLP(bs=1, n_dim=dim, n_equations=1, n_auxiliary=0, n_step=n_step, step_size=0.1, order=2, n_iv=1, device='cpu', step_list=_steps)

        derivative_constraints,deriv_values = ode.build_derivative_tensor(steps)
        #eq_constraints = self.ode.build_equation_tensor(coeffs)

        fix_values = ode.value_dict[ConstraintType.Derivative]

        print('A',deriv_values)
        print('B', fix_values)

        #print(ode.value_dict[ConstraintType.Derivative])
        diff = deriv_values - torch.tensor(fix_values)
        print(diff)
        print(diff.mean())

if __name__=="__main__":
    #ODESYSLP().ode()
    test()
