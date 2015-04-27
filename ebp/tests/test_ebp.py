import numpy as np
import numpy.testing as npt
import ebp


def test_gaussian_kernel():
    xx3d = np.array(np.meshgrid(np.arange(-10,10,1),
                              np.arange(-10,10,1),
                              np.arange(-10,10,1)))

    # Means are identical:
    mean3d = [0, 0, 0]
    # Variance/covariance matrix:
    sigma3d = np.array([[10, 10, 0], [0,10, 10], [10,10,10]])

    gauss3d = ebp.gaussian_kernel(xx3d, np.hstack([mean3d, sigma3d.ravel()]))

    # Verify against a 1D case:
    x = xx3d[0,0,:,0]
    gauss1d = ( 1/(sigma3d[0,0] * np.sqrt(np.pi))
                * np.exp(-0.5 * (x - mean3d[0]) **2 / sigma3d[0,0]))

    # should be identical up to a scaling factor:
    ratio = gauss3d[10, 10]/gauss1d
    should_be_ones = ratio / np.mean(ratio)
    npt.assert_almost_equal(should_be_ones, np.ones(should_be_ones.shape))

    # Test that the inputs are being checked:
    npt.assert_raises(ValueError, ebp.gaussian_kernel, xx3d, mean3d)
    
def test_gaussian_oracle():
    mean1 = [20, 20]
    sigma1 = [10, 10]
    params1 = np.hstack([mean1, sigma1])

    xx2d = np.array(np.meshgrid(np.arange(-100, 100, 5),
                                np.arange(-100, 100, 5))).reshape(2, -1)

    gauss2d = ebp.gaussian_kernel(xx2d, params1)
    # Oh, we're totally stuffing this ballot box:
    initial  = params1 
    theta_est = ebp.leastsq_oracle(xx2d, gauss2d, ebp.gaussian_kernel, initial,
                               bounds=[[None, None], [None, None], [0, None],
                                       [0, None]]) # Variance can't be negative!

    npt.assert_almost_equal(theta_est, params1)
                                

def test_mixture_of_kernels():

    for xx2d in [np.array(np.meshgrid(np.arange(-100, 100, 5),
                                      np.arange(-100, 100, 5))),

                 np.array(np.meshgrid(np.arange(-100, 100, 5),
                                      np.arange(-100, 100, 5))).reshape(2, -1)]:

        # Make a 2D mixture of 2 Gaussians:
        mean1 = [20, 20]
        sigma1 = [10, 10]
        params1 = np.hstack([mean1, sigma1])
        kernel1 = ebp.gaussian_kernel(xx2d, params1)
        mean2 = [30, 40]
        sigma2 = [10, 50]
        params2 = np.hstack([mean2, sigma2])
        kernel2 = ebp.gaussian_kernel(xx2d, params2)
        betas = [0.3, 0.7]
        signal1 = betas[0] * kernel1 + betas[1] * kernel2

        signal2 = ebp.mixture_of_kernels(xx2d, betas, [params1, params2],
                                         ebp.gaussian_kernel)

        npt.assert_almost_equal(signal1, signal2)

def test_kernel_error():
    """

    """
    for xx2d in [np.array(np.meshgrid(np.arange(-100, 100, 5),
                                      np.arange(-100, 100, 5))),

                 np.array(np.meshgrid(np.arange(-100, 100, 5),
                                      np.arange(-100, 100, 5))).reshape(2, -1)]:

        mean1 = [20, 20]
        sigma1 = [10, 10]
        params1 = np.hstack([mean1, sigma1])
        mean2 = [30, 40]
        sigma2 = [10, 50]
        params2 = np.hstack([mean2, sigma2])

        params = [params1, params2]
        betas = [0.3, 0.7]
        
        y = ebp.mixture_of_kernels(xx2d, betas, params,
                                   ebp.gaussian_kernel)

        err = ebp.kernel_err(y, xx2d, betas, params, ebp.gaussian_kernel)

        npt.assert_almost_equal(err, np.zeros(err.shape))

def test_parameters_to_regressors():
    """

    """

    xx2d = np.array(np.meshgrid(np.arange(-100, 100, 5),
                                np.arange(-100, 100, 5)))
    mean1 = [20, 20]
    sigma1 = [10, 10]
    params1 = np.hstack([mean1, sigma1])
    mean2 = [30, 40]
    sigma2 = [10, 50]
    params2 = np.hstack([mean2, sigma2])
    params = [params1, params2]

    reference = np.array([ebp.gaussian_kernel(xx2d.reshape(2, -1), p)
                          for p in params]).T

    regressors = ebp.parameters_to_regressors(xx2d, ebp.gaussian_kernel, params)

    npt.assert_almost_equal(reference, regressors)

    
def test_solve_nnls():
    """

    """
    regressor_params = []
    for mean1 in range(10, 50, 10):
        for mean2 in range(10, 50, 10):
            for sigma1 in range(10, 50, 10):
                for sigma2 in range(10, 50, 10):
                    regressor_params.append([mean1, mean2, sigma1, sigma2])

    mean1 = [20, 20]
    sigma1 = [10, 10]
    params1 = np.hstack([mean1, sigma1])
    mean2 = [30, 40]
    sigma2 = [10, 50]
    params2 = np.hstack([mean2, sigma2])

    params = [params1, params2]
    betas = [0.3, 0.7]

    for xx2d in [np.array(np.meshgrid(np.arange(-100, 100, 5),
                                      np.arange(-100, 100, 5))),

                 np.array(np.meshgrid(np.arange(-100, 100, 5),
                                      np.arange(-100, 100, 5))).reshape(2, -1)]:

            y = ebp.mixture_of_kernels(xx2d, betas, params,
                                   ebp.gaussian_kernel)

            beta_hat, rnorm = ebp.solve_nnls(xx2d, y, ebp.gaussian_kernel,
                                             regressor_params)

            design = ebp.parameters_to_regressors(xx2d,
                                                  ebp.gaussian_kernel,
                                                  regressor_params)
            y_hat = np.dot(design, beta_hat)

            npt.assert_almost_equal(y.ravel(), y_hat, decimal=2)
            

def test_elastic_basis_pursuit():
    """
    
    """
    
    xx2d = np.array(np.meshgrid(np.arange(0,50,1),
                                np.arange(0,50,1)))
    bounds = [[0,50], [0,50], [0, None], [0, None]]
    mean1 = [20, 20]
    sigma1 = [10, 10]
    params1 = np.hstack([mean1, sigma1])

    kernel1 = ebp.gaussian_kernel(xx2d, params1)
    mean2 = [30, 40]
    sigma2 = [10, 50]
    params2 = np.hstack([mean2, sigma2])
    kernel2 = ebp.gaussian_kernel(xx2d, params2)
    beta = [0.3, 0.7]
    signal = (beta[0] * kernel1 + beta[1] * kernel2).ravel()
    initial_theta = np.hstack([np.mean(xx2d, axis=(-1, -2)), 1, 1])

    theta, err, r = ebp.elastic_basis_pursuit(xx2d.reshape(-1, signal.shape[0]),
                                              signal,
                                              ebp.leastsq_oracle,
                                              ebp.gaussian_kernel,
                                              initial_theta=initial_theta,
                                              bounds=bounds,
                                              max_iter=1000,
                                              beta_tol=10e-6)


