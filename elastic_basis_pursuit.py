"""
Elastic basis pursuit
"""

import numpy as np
import numpy.linalg as nla
import leastsqbound as lsq
import sklearn.linear_model as lm

def err_func(params, x, y, func):
        """
        Error function for fitting a function
        
        Parameters
        ----------
        params : tuple
            A tuple with the parameters of `func` according to their order of
            input

        x : float array 
            An independent variable. 
        
        y : float array
            The dependent variable. 
        
        func : function
            A function with inputs: `(x, *params)`
        
        Returns
        -------
        The sum of squared marginals of the fit to x/y given the params
        """
        # We ravel both, so that we can accomodate multi-d input without having
        # to think about it:
        return np.ravel(y) - np.ravel(func(x, params))


def gaussian_kernel(x, params):
    """
    A multi-dimensional Gaussian kernel function

    Useful for creating and testing EBP with simple Gaussian Mixture Models

    Parameters
    ----------
    x : ndarray
        The independent variable over which the Gaussian is calculated

    params : ndarray
       If this is a 1D array, it could have one of few things:

       [mu_1, mu_2, ... mu_n, sigma_1, sigma_2, ... sigma_n]

       Or:
       [mu_1, mu_2, ... mu_n, var_covar_matrix]

       where: 

       var_covar_matrix needs to be reshaped into n-by-n 
       
    """
    if len(params) == x.shape[0] * 2:
        mu = params[:x.shape[0]]
        sigma = np.diag(params[x.shape[0]:])
    elif len(params) == x.shape[0] + x.shape[0] ** 2:
        mu = params[:x.shape[0]]
        sigma = np.reshape(params[x.shape[0]:], (x.shape[0], x.shape[0]))
    else:
        e_s = "Inputs to gaussian_kernel don't have the right dimensions"
        raise ValueError(e_s)

    dims = mu.shape[0]
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


def leastsq_oracle(x, y, func, initial, bounds=None):
    """
    This is a generic oracle function that uses bounded least squares to find
    the parameters in each iteration of EBP, and requires initial parameters. 

    Parameters
    ----------
    x : ndarray
        Input to the kernel function.
    y : ndarray
        Data to fit to.
    func : callalble
       The kernel function to be specified by this oracle.
    initial : list/array
        initial setting for the parameters of the function. This has to be
        something that func knows what to do with.
    """    
    return lsq.leastsqbound(err_func, initial, args=(x, y, func),
                            bounds=bounds)[0]


def mixture_of_kernels(x, betas, params, kernel):
    """

    Generate the signal from a mixture of kernels

    Parameters
    ----------
    x : ndarray

    betas : 1D array
        Coefficients for the linear summation of the kernels

    params : list
        A set of parameters for each one of the kernels 

    kernel : callable
    
    """
    betas = np.asarray(betas)
    out = np.zeros(np.prod(x.shape[1:]))

    for i in xrange(betas.shape[0]):
        out += np.dot(betas[i], kernel(x, params[i]))

    return out

def kernel_err(y, x, betas, params, kernel):
    """
    An error function for a mixture of kernels, each one parameterized by its
    own set of params, and weighted by a beta


    Note
    ----
    For a given set of betas, params, this can be used as a within set error
    function, or to estimate the cross-validation error against another set of
    y, x values, sub-sampled from the whole original set, or from a left-out
    portion
    """
    return y - mixture_of_kernels(x, betas, params, kernel)
    
     
def elastic_basis_pursuit(x, y, oracle, func):
    """
    Elastic basis pursuit

    Fit a mixture model::

    ..math::

      y = \sum{w_i f_{\theta_i} (x_i)}

    with y data, f a kernel function parameterized by $\theta_i$ and \w_i a
    non-negative weight, and x inputs to the kernel function

    Parameters
    ----------
    x : 1D/2D array
        The independent variable that produces the data

    y : 1D/2D darray
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
    r = y

    # Find the first function:
    theta0 = oracle(x, y)
    active_set = (func(x, y, theta0))

    while np.sum(r**2>tol):

        
        r = active_set[-1] - y
        
        regressors = np.hstack(active_set)
    

