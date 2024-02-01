#Batched sparse conjugate gradient adapted from cupy cg

import torch
import ipdb

def block_mv(A, x):
    """shape x: (b, d), A sparse block"""
    b = x.shape[0]
    x = x.reshape(-1)

    y = torch.mv(A, x)
    y = y.reshape(b, -1)
    return y

def cg_block(A, b, x0=None, tol=1e-12, maxiter=None, M=None, callback=None,
       atol=None):
    """Uses Conjugate Gradient iteration to solve ``Ax = b``.

        A block sparse, b: (b, d)
    Args:
        A (ndarray, spmatrix or LinearOperator): The real or complex matrix of
            the linear system with shape ``(n, n)``. ``A`` must be a hermitian,
            positive definitive matrix with type of :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        b (cupy.ndarray): Right hand side of the linear system with shape
            ``(n,)`` or ``(n, 1)``.
        x0 (cupy.ndarray): Starting guess for the solution.
        tol (float): Tolerance for convergence.
        maxiter (int): Maximum number of iterations.
        M (ndarray, spmatrix or LinearOperator): Preconditioner for ``A``.
            The preconditioner should approximate the inverse of ``A``.
            ``M`` must be :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        callback (function): User-specified function to call after each
            iteration. It is called as ``callback(xk)``, where ``xk`` is the
            current solution vector.
        atol (float): Tolerance for convergence.

    Returns:
        tuple:
            It returns ``x`` (cupy.ndarray) and ``info`` (int) where ``x`` is
            the converged solution and ``info`` provides convergence
            information.

    .. seealso:: :func:`scipy.sparse.linalg.cg`
    """
    #A, M, x, b = _make_system(A, M, x0, b)
    #matvec = A.matvec
    #psolve = M.matvec

    #n = A.shape[0]
    #n = A.shape[1]
    #if maxiter is None:
    #    maxiter = n * 10
    #if n == 0:
    #    return cupy.empty_like(b), 0
    #TODO fix this. check all norms and use masks 
    #b_norm = torch.linalg.norm(b[0])
    #b_norm = torch.linalg.vector_norm(b[0],dim=-1)
    b_norm = torch.linalg.vector_norm(b,dim=-1)
    #if b_norm == 0:
    cont_mask = (b_norm>1e-9)
    #if b_norm < 1e-10:
    if not cont_mask.any():
        print('zero return')
        return b, 0

    cont_mask = cont_mask.float()
    if atol is None:
        #atol = tol * float(b_norm)
        atol = tol * b_norm
    else:
        atol = max(float(atol), tol * float(b_norm))
        #atol = float(atol)

    #r = b - matvec(x)
    #b = b.unsqueeze(-1)
    x = torch.zeros_like(b)
    #r = b - torch.bmm(A,x)#.reshape(b.shape)
    r = b - block_mv(A,x)#.reshape(b.shape)
    iters = 0
    rho = 0
    #ipdb.set_trace()
    while iters < maxiter:
        #z = psolve(r)
        z = r #psolve(r)
        rho1 = rho
        #rho = cublas.dotc(r, z)
        rho = (r*z).sum(dim=1)
        if iters == 0:
            p = z
        else:
            beta = rho / rho1
            beta = torch.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)

            beta = beta.unsqueeze(1)
            p = z + beta * p
        #q = matvec(p)
        #q = torch.bmm(A, p)
        #q = bmm_fix(A, p)
        q = block_mv(A, p)
        #alpha = rho / cublas.dotc(p, q)
        alpha = rho / (p*q).sum(dim=1)

        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        alpha = alpha*cont_mask

        alpha = alpha.unsqueeze(1)
        x = x + alpha * p
        r = r - alpha * q
        iters += 1
        if callback is not None:
            callback(x)
        #resid = cublas.nrm2(r)
        #resid = cublas.nrm2(r)
        resid = torch.linalg.vector_norm(r, dim=1)
        res_mask = (resid > atol).float()
        cont_mask = cont_mask*res_mask
        #if resid.max() <= atol:
        #    break

    info = 0
    #if iters == maxiter and not (resid.max() <= atol):
    #    info = iters

    return x, info

def bmm_fix(A, x):
    b = A.shape[0]
    r = [A[i]@x[i] for i in range(b)]
    r = torch.stack(r, dim=0)
    return r


def cg(A, b, x0=None, tol=1e-5, maxiter=None, M=None, callback=None,
       atol=None):
    """Uses Conjugate Gradient iteration to solve ``Ax = b``.

    Args:
        A (ndarray, spmatrix or LinearOperator): The real or complex matrix of
            the linear system with shape ``(n, n)``. ``A`` must be a hermitian,
            positive definitive matrix with type of :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        b (cupy.ndarray): Right hand side of the linear system with shape
            ``(n,)`` or ``(n, 1)``.
        x0 (cupy.ndarray): Starting guess for the solution.
        tol (float): Tolerance for convergence.
        maxiter (int): Maximum number of iterations.
        M (ndarray, spmatrix or LinearOperator): Preconditioner for ``A``.
            The preconditioner should approximate the inverse of ``A``.
            ``M`` must be :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        callback (function): User-specified function to call after each
            iteration. It is called as ``callback(xk)``, where ``xk`` is the
            current solution vector.
        atol (float): Tolerance for convergence.

    Returns:
        tuple:
            It returns ``x`` (cupy.ndarray) and ``info`` (int) where ``x`` is
            the converged solution and ``info`` provides convergence
            information.

    .. seealso:: :func:`scipy.sparse.linalg.cg`
    """
    #A, M, x, b = _make_system(A, M, x0, b)
    #matvec = A.matvec
    #psolve = M.matvec

    #n = A.shape[0]
    n = A.shape[1]
    if maxiter is None:
        maxiter = n * 10
    #if n == 0:
    #    return cupy.empty_like(b), 0
    #TODO fix this. check all norms and use masks 
    #b_norm = torch.linalg.norm(b[0])
    b_norm = torch.linalg.vector_norm(b[0],dim=-1)
    #if b_norm == 0:
    if b_norm < 1e-8:
        return b, 0
    if atol is None:
        atol = tol * float(b_norm)
    else:
        atol = max(float(atol), tol * float(b_norm))
        #atol = float(atol)

    #r = b - matvec(x)
    b = b.unsqueeze(-1)
    x = torch.zeros_like(b)
    #r = b - torch.bmm(A,x)#.reshape(b.shape)
    r = b - bmm_fix(A,x)#.reshape(b.shape)
    iters = 0
    rho = 0
    #ipdb.set_trace()
    while iters < maxiter:
        #z = psolve(r)
        z = r #psolve(r)
        rho1 = rho
        #rho = cublas.dotc(r, z)
        rho = (r*z).sum(dim=1)
        if iters == 0:
            p = z
        else:
            beta = rho / rho1
            beta = beta.unsqueeze(1)
            p = z + beta * p
        #q = matvec(p)
        #q = torch.bmm(A, p)
        q = bmm_fix(A, p)
        #alpha = rho / cublas.dotc(p, q)
        alpha = rho / (p*q).sum(dim=1)
        alpha = alpha.unsqueeze(1)
        x = x + alpha * p
        r = r - alpha * q
        iters += 1
        if callback is not None:
            callback(x)
        #resid = cublas.nrm2(r)
        #resid = cublas.nrm2(r)
        resid = torch.linalg.vector_norm(r, dim=(1,2))
        if resid.max() <= atol:
            break

    info = 0
    if iters == maxiter and not (resid.max() <= atol):
        info = iters

    return x.squeeze(-1), info