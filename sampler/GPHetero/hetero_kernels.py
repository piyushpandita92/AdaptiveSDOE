from __future__ import print_function, absolute_import
from functools import reduce

import tensorflow as tf
import numpy as np
import gpflow
from gpflow.param import Param, Parameterized, AutoFlow
from gpflow.kernels import Kern
from gpflow import transforms
from gpflow._settings import settings
from gpflow.quadrature import hermgauss, mvhermgauss, mvnquad

float_type = settings.dtypes.float_type
int_type = settings.dtypes.int_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64


class Stationary(Kern):
    """
    Base class for kernels that are stationary, that is, they only depend on

        r = || x - x' ||

    This class handles 'ARD' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale).
    """

    def __init__(self, input_dim, variance=1.0, lengthscales=None,
                 active_dims=None, ARD=False):
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter
        - lengthscales is the initial value for the lengthscales parameter
          defaults to 1.0 (ARD=False) or np.ones(input_dim) (ARD=True).
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        - ARD specifies whether the kernel has one lengthscale per dimension
          (ARD=True) or a single lengthscale (ARD=False).
        """
        Kern.__init__(self, input_dim, active_dims)
        self.scoped_keys.extend(['square_dist', 'euclid_dist'])
        self.variance = Param(variance, transforms.positive)
        if ARD:
            if lengthscales is None:
                lengthscales = np.ones(input_dim, np_float_type)
            else:
                # accepts float or array:
                lengthscales = lengthscales * np.ones(input_dim, np_float_type)
            self.lengthscales = Param(lengthscales, transforms.positive)
            self.ARD = True
        else:
            if lengthscales is None:
                lengthscales = 1.0
            self.lengthscales = Param(lengthscales, transforms.positive)
            self.ARD = False

    def square_dist(self, X, X2):
        X = X / self.lengthscales
        Xs = tf.reduce_sum(tf.square(X), 1)
        if X2 is None:
            return -2 * tf.matmul(X, X, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        else:
            X2 = X2 / self.lengthscales
            X2s = tf.reduce_sum(tf.square(X2), 1)
            return -2 * tf.matmul(X, X2, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return tf.sqrt(r2 + 1e-12)

    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))

class RBF(Stationary):
    """
    The radial basis function (RBF) or squared exponential kernel
    """
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        return self.variance * tf.exp(-self.square_dist(X, X2) / 2)
    
    @AutoFlow((float_type, [None, None]), (float_type, [None, None]),
              (float_type, [None, None]), (float_type, [None, None]))

    def compute_K(self, X1, X2):
        return self.K(X1, X2)

class NonStationaryLengthscaleRBF(Kern):
    """
    Non-stationary 1D RBF kernel
    For more info refer to paper:
    https://arxiv.org/abs/1508.04319
    """
    def __init__(self):
        Kern.__init__(self, input_dim = 1, active_dims= [0])
        self.signal_variance = Param(1.0, transform=transforms.positive)
        
    def K(self, X1, Lexp1, X2, Lexp2):
        """
        X1, X2 : input points
        Lexp1 and Sexp1 are exponential of latent GPs 
        L1(.) representing log of non-stationary lengthscale values at points X1 and
        S1(.) representing log of non-stationary signal variance values at points X1.
        """
        dist_sqr = tf.square(X1 - tf.transpose(X2))
        l_sqr = tf.square(Lexp1) + tf.square(tf.transpose(Lexp2))
        l_2_prod = 2. * Lexp1 * tf.transpose(Lexp2)
        # var_prod = self.signal_variance
        cov = tf.sqrt(l_2_prod / l_sqr) * tf.exp(-1. * dist_sqr / l_sqr)
        return cov
    
    @AutoFlow((float_type, [None, None]), (float_type, [None, None]),
              (float_type, [None, None]), (float_type, [None, None]))
    def compute_K(self, X1, Lexp1, X2, Lexp2):
        return self.K(X1, Lexp1, X2, Lexp2)

class NonStationaryRBF(Kern):
    """
    Non-stationary 1D RBF kernel
    For more info refer to paper:
        https://arxiv.org/abs/1508.04319
    """
    def __init__(self):
        Kern.__init__(self, input_dim = 1, active_dims= [0])
        
    def K(self, X1, Lexp1, Sexp1, X2, Lexp2, Sexp2):
        """
        X1, X2 : input points
        Lexp1 and Sexp1 are exponential of latent GPs 
        L1(.) representing log of non-stationary lengthscale values at points X1 and
        S1(.) representing log of non-stationary signal variance values at points X1.
        """
        dist_sqr = tf.square(X1 - tf.transpose(X2))
        l_sqr = tf.square(Lexp1) + tf.square(tf.transpose(Lexp2))
        l_2_prod = 2 * Lexp1 * tf.transpose(Lexp2)
        var_prod = Sexp1 * tf.transpose(Sexp2)
        cov = var_prod * tf.sqrt(l_2_prod / l_sqr) * tf.exp(-1 * dist_sqr / l_sqr)
        return cov
    
    @AutoFlow((float_type, [None, None]), (float_type, [None, None]),
              (float_type, [None, None]), (float_type, [None, None]),
              (float_type, [None, None]), (float_type, [None, None]))
    def compute_K(self, X1, Lexp1, Sexp1, X2, Lexp2, Sexp2):
        return self.K(X1, Lexp1, Sexp1, X2, Lexp2, Sexp2)


if __name__ == '__main__':
    import numpy as np
    A = np.arange(2, 100)[:,None]
    B = np.arange(1, 100)[:,None]
    C = np.arange(1, 100)[:,None]
    Cov = NonStationaryLengthscaleRBF()
    #r = Cov.compute_K(A,A,A,A)
    X = np.random.rand(3, 2)
    #K = gpflow.kernels.RBF(input_dim = 2, ARD = True)
    a = Cov.compute_K(A, A, A, A)
    Cov = NonStatLRBFMultiD()
    #r = Cov.compute_K(A,A,A,A)
    X = np.random.rand(3, 2)
    Cov.compute_Ka(X, X, X, X)
    a = Cov.compute_K(X, X, X, X)