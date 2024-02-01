#adapted from https://github.com/bstollnitz/sindy/tree/main/sindy/lorenz-custom/src/common.py

import numpy as np
import torch

def create_library_tensor_batched(u: torch.tensor, polynomial_order: int,
                   use_trig: bool, constant=False) -> np.ndarray:
    """Creates a matrix containing a library of candidate functions.

    For example, if our u depends on x, y, and z, and we specify
    polynomial_order=2 and use_trig=false, our terms would be:
    1, x, y, z, x^2, xy, xz, y^2, yz, z^2.
    """
    var_list = []
    (b, m, n) = u.shape
    if constant:
        theta = torch.ones((b,m, 1)).type_as(u)
        #var_list.append('1')

        # Polynomials of order 1.
        theta = torch.cat((theta, u),dim=-1)
    else:
        theta = u
    #for i in range(n):
    #    var_list.append(f'x{i}')

    # Polynomials of order 2.
    if polynomial_order >= 2:
        for i in range(n):
            for j in range(i, n):
                theta = torch.cat((theta, u[:,:, i:i + 1] * u[:,:, j:j + 1]), dim=-1)
                #var_list.append(f'x{i}*x{j}')

    # Polynomials of order 3.
    if polynomial_order >= 3:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    theta = torch.cat(
                        (theta, u[:,:, i:i + 1] * u[:,:, j:j + 1] * u[:,:, k:k + 1]), dim=-1)

                    #var_list.append(f'x{i}*x{j}*x{k}')

    # Polynomials of order 4.
    if polynomial_order >= 4:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    for l in range(k, n):
                        theta = torch.cat(
                            (theta, u[:,:, i:i + 1] * u[:,:, j:j + 1] *
                             u[:,:, k:k + 1] * u[:,:, l:l + 1]), dim=-1)
                        #var_list.append(f'x{i}*x{j}*x{k}*x{l}')

    # Polynomials of order 5.
    if polynomial_order >= 5:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    for l in range(k, n):
                        for m in range(l, n):
                            theta = torch.cat(
                                (theta, u[:,:, i:i + 1] * u[:,:, j:j + 1] *
                                 u[:,:, k:k + 1] * u[:,:, l:l + 1] * u[:,:, m:m + 1]), dim=-1)
                            #var_list.append(f'x{i}*x{j}*x{k}*x{l}*x{m}')

    if use_trig:
        for i in range(1, 11):
            theta = torch.hstack((theta, np.sin(i * u), np.cos(i * u)))

    return theta, var_list


def create_library_tensor(u: torch.tensor, polynomial_order: int,
                   use_trig: bool, constant=False) -> np.ndarray:
    """Creates a matrix containing a library of candidate functions.

    For example, if our u depends on x, y, and z, and we specify
    polynomial_order=2 and use_trig=false, our terms would be:
    1, x, y, z, x^2, xy, xz, y^2, yz, z^2.
    """
    var_list = []
    (m, n) = u.shape
    if constant:
        theta = torch.ones((m, 1)).type_as(u)
        var_list.append('1')

        # Polynomials of order 1.
        theta = torch.hstack((theta, u))
    else:
        theta = u
    for i in range(n):
        var_list.append(f'x{i}')

    # Polynomials of order 2.
    if polynomial_order >= 2:
        for i in range(n):
            for j in range(i, n):
                theta = torch.hstack((theta, u[:, i:i + 1] * u[:, j:j + 1]))
                var_list.append(f'x{i}*x{j}')

    # Polynomials of order 3.
    if polynomial_order >= 3:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    theta = torch.hstack(
                        (theta, u[:, i:i + 1] * u[:, j:j + 1] * u[:, k:k + 1]))

                    var_list.append(f'x{i}*x{j}*x{k}')

    # Polynomials of order 4.
    if polynomial_order >= 4:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    for l in range(k, n):
                        theta = torch.hstack(
                            (theta, u[:, i:i + 1] * u[:, j:j + 1] *
                             u[:, k:k + 1] * u[:, l:l + 1]))
                        var_list.append(f'x{i}*x{j}*x{k}*x{l}')

    # Polynomials of order 5.
    if polynomial_order >= 5:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    for l in range(k, n):
                        for m in range(l, n):
                            theta = torch.hstack(
                                (theta, u[:, i:i + 1] * u[:, j:j + 1] *
                                 u[:, k:k + 1] * u[:, l:l + 1] * u[:, m:m + 1]))
                            var_list.append(f'x{i}*x{j}*x{k}*x{l}*x{m}')

    if use_trig:
        for i in range(1, 11):
            theta = torch.hstack((theta, np.sin(i * u), np.cos(i * u)))

    return theta, var_list


def create_library(u: np.ndarray, polynomial_order: int,
                   use_trig: bool, constant=False) -> np.ndarray:
    """Creates a matrix containing a library of candidate functions.

    For example, if our u depends on x, y, and z, and we specify
    polynomial_order=2 and use_trig=false, our terms would be:
    1, x, y, z, x^2, xy, xz, y^2, yz, z^2.
    """
    var_list = []
    (m, n) = u.shape
    if constant:
        theta = np.ones((m, 1))
        var_list.append('1')

        # Polynomials of order 1.
        theta = np.hstack((theta, u))
    else:
        theta = u
    for i in range(n):
        var_list.append(f'x{i}')

    # Polynomials of order 2.
    if polynomial_order >= 2:
        for i in range(n):
            for j in range(i, n):
                theta = np.hstack((theta, u[:, i:i + 1] * u[:, j:j + 1]))
                var_list.append(f'x{i}*x{j}')

    # Polynomials of order 3.
    if polynomial_order >= 3:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    theta = np.hstack(
                        (theta, u[:, i:i + 1] * u[:, j:j + 1] * u[:, k:k + 1]))

                    var_list.append(f'x{i}*x{j}*x{k}')

    # Polynomials of order 4.
    if polynomial_order >= 4:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    for l in range(k, n):
                        theta = np.hstack(
                            (theta, u[:, i:i + 1] * u[:, j:j + 1] *
                             u[:, k:k + 1] * u[:, l:l + 1]))
                        var_list.append(f'x{i}*x{j}*x{k}*x{l}')

    # Polynomials of order 5.
    if polynomial_order >= 5:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    for l in range(k, n):
                        for m in range(l, n):
                            theta = np.hstack(
                                (theta, u[:, i:i + 1] * u[:, j:j + 1] *
                                 u[:, k:k + 1] * u[:, l:l + 1] * u[:, m:m + 1]))
                            var_list.append(f'x{i}*x{j}*x{k}*x{l}*x{m}')

    if use_trig:
        for i in range(1, 11):
            theta = np.hstack((theta, np.sin(i * u), np.cos(i * u)))

    return theta, var_list


def basis_repr(coeffs, basis_vars):
    coeffs = coeffs.detach().squeeze()
    n_basis = len(basis_vars)
    #print(coeffs.shape[0])
    assert(coeffs.shape[0] == n_basis)
    dim = coeffs.shape[1]

    repr = dict()
    for i in range(dim):
        ii = str(i)
        key = 'dx'+ii
        repr[key] = "0 "

        for j in range(n_basis):
            cc = coeffs[j,i].item()
            repr[key] += f' + {cc:.4f}*{basis_vars[j]}'
    return repr