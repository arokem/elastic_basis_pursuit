"""
Elastic basis pursuit
"""

import numpy as np
import numpy.linalg as nla

def elastic_basis_pursuit(x, y, oracle, func, p_xval=0.1):
    """
    Elastic basis pursuit

    Fit a mixture model::

    ..math::

      y = \sum{w_i f_{\theta_i} (x_i)}

    with y data, f a kernel function parameterized by $\theta_i$ and \w_i a
    non-negative weight, and x inputs to the kernel function

    Parameters
    ----------
    x : ndarray
        The independent variable that produces the data

    y : ndarray
        The data to be fit.

    oracle : callable

        This is a function that takes data (`y`) and an input function (`func`)
        and returns another callable that when called with the i-th item in x
        will produce something that has the dimensions of the i-th item in y.

    func : callable
        A skeleton for the oracle function to optimize. Must take something
        of the dimensions of x (together with params, and with args) and return
        something of the dimensions of y.  
        
    """




def gaussian_kernel(x, *params):
    """
    A multi-dimensional Gaussian kernel function

    Useful for creating and testing EBP with simple Gaussian Mixture Models

    Parameters
    ----------
    x : ndarray
        The independent variable over which the Gaussian is calculated

    """
    mu = np.asarray(params[0])
    sigma = np.asarray(params[1])
    if mu.shape[0] != sigma.shape[0]:
        raise ValueError("Inputs mu and sigma should have the same dimensions")
    else:
        dims = mu.shape[0]
        
    if len(sigma.shape) == 1:
        sigma = np.diag(sigma)
    elif len(sigma.shape) == 2:
        pass
    else:
        e_s = "Input sigma should be a 1D array (for diagonal "
        e_s += "covariance matrix), or 2D (for non-diagonal covariance) matrix"
        raise ValueError(e_s)

    while len(mu.shape) < len(x.shape): 
        mu = mu[..., None]

    shape_tuple = x.shape[1:]
    diff = (x - mu).reshape(x.shape[0], -1)
    sigma_inv = nla.inv(sigma)
    mult1 = np.dot(diff.T, sigma_inv)
    mult2 =  (np.diag(np.dot(mult1, diff))).reshape(shape_tuple)
    norm_factor = 1/(np.sqrt((2*np.pi)**dims * nla.det(sigma)))
    gauss = norm_factor * np.exp(-0.5 * mult2) 
    return gauss
