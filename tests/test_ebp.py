import numpy as np
import numpy.testing as npt
import elastic_basis_pursuit as ebp

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

    xx2d = np.array(np.meshgrid(np.arange(-100, 100, 5),
                                np.arange(-100, 100, 5))).reshape(2, -1)

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
