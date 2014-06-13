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

    gauss3d = ebp.gaussian_kernel(xx3d, mean3d, sigma3d)

    # Verify against a 1D case:
    x = xx3d[0,0,:,0]
    gauss1d = ( 1/(sigma3d[0,0] * np.sqrt(np.pi))
                * np.exp(-0.5 * (x - mean3d[0]) **2 / sigma3d[0,0]))

    # should be identical up to a scaling factor:

    ratio = gauss3d[10,10]/gauss1d
    should_be_ones = ratio / np.mean(ratio)
    npt.assert_almost_equal(should_be_ones, np.ones(should_be_ones.shape))
    
    
