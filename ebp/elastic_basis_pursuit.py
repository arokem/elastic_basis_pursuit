"""
Elastic basis pursuit
"""

import numpy as np
import numpy.linalg as nla
import leastsqbound as lsq
import sklearn.linear_model as lm
import scipy.optimize as opt


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
    mu = np.asarray(params[:x.shape[0]])
    if len(params) == x.shape[0] * 2:
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


def leastsq_oracle(x, y, kernel, initial=None, bounds=None):
    """
    This is a generic oracle function that uses bounded least squares to find
    the parameters in each iteration of EBP, and requires initial parameters. 

    Parameters
    ----------
    x : ndarray
        Input to the kernel function.
    y : ndarray
        Data to fit to.
    kernel : callalble
       The kernel function to be specified by this oracle.
    initial : list/array
        initial setting for the parameters of the function. This has to be
        something that kernel knows what to do with.
    """
    return lsq.leastsqbound(err_func, initial, args=(x, y, kernel),
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
    out = np.zeros(x.shape[1:])

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

    
def parameters_to_regressors(x, kernel, params):
    """
    Maps from parameters to regressors through the kernel function

    Parameters
    ----------
    x : ndarray
        Input
    kernel : callable
       The kernel function
    params : list
       The parameters for each one of the kernel functions
    
    """
    # Ravel the secondary dimensions of this:
    x = x.reshape(x.shape[0], -1).squeeze()
    regressors = np.zeros((len(params), x.shape[-1]))
    for i, p in enumerate(params):
        regressors[i] = kernel(x, p)
    return regressors.T
    

def solve_nnls(x, y, kernel=None, params=None, design=None):
    """
    Solve the mixture problem using NNLS

    Parameters
    ----------
    x : ndarray
    y : ndarray

    kernel : callable
    params : list

    """
    if design is None and (kernel is None or params is None):
        e_s = "Need to provide either design matrix, or kernel and list of"
        e_s += "params for generating the design matrix"
        raise ValueError(e_s)

    if design is None:
        A = parameters_to_regressors(x, kernel, params)
    else:
        A = design
    y = y.ravel()
    beta_hat, rnorm = opt.nnls(A, y)
    return beta_hat, rnorm
    
    
def elastic_basis_pursuit(x, y, oracle, kernel, initial_theta=None, bounds=None,
                          max_iter=1000, beta_tol=10e-6, xval=True):
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
        This is a function that takes data (`x`/`y`) and a kernel function
        (`kernel`) and returns the params theta for the kernel given x and
        y. The oracle can use any optimization routine, and any cost function

    kernel : callable
        A skeleton for the oracle function to optimize. Must take something
        of the dimensions of x (together with params, and with args) and return
        something of the dimensions of y.  

    initial_theta : list/array
        The initial parameter guess

    bounds : the bounds on 
    """
    fit_x = x[..., ::2]
    validate_x = x[..., 1::2]
    fit_y = y[::2]
    validate_y = y[1::2]

    # Initialize a bunch of empty lists to hold the state:
    theta = []
    est = [] 
    design_list = []
    r = []
    err = [np.dot(fit_y, fit_y)] # Start with the assumption of 
    err_norm = []
    # Initialize the residuals with the fit_data:
    r.append(fit_y)

    # Limit this by number of iterations
    for i in range(max_iter):
        theta.append(oracle(fit_x, r[-1], kernel, initial_theta,
			    bounds=bounds))
        design = parameters_to_regressors(fit_x, kernel, theta)
        beta_hat, rnorm = solve_nnls(fit_x, fit_y, design=design)
        # Here comes the "elastic" bit. We exclude kernels with insignificant
        # contributions: 
        keep_idx = np.where(beta_hat > beta_tol)
        # We want this to still be a list (so we can 'append'):
        theta = list(np.array(theta)[keep_idx])
        beta_hat = beta_hat[keep_idx]
        design = design[:, keep_idx[0]]
        # Move on with the shrunken basis set:
        est.append(np.dot(design, beta_hat))
        r.append(fit_y - est[-1])
        # Cross-validation:
        xval_design = parameters_to_regressors(validate_x, kernel, theta)
        xval_est = np.dot(xval_design, beta_hat)
        xval_r = validate_y - xval_est
        err.append(np.dot(xval_r, xval_r))
        # If error just grew, we bail:
        if err[i+1] > err[i]:
            break
	
    return theta, err, r
