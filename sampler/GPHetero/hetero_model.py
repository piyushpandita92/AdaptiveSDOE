import numpy as np
from .hetero_likelihoods import GaussianHeteroNoise, Gaussian
import gpflow
from gpflow.param import Param
from gpflow import transforms
from gpflow.model import Model
from gpflow.mean_functions import Zero
import tensorflow as tf
from gpflow.param import AutoFlow, DataHolder, ParamList
from gpflow._settings import settings
float_type = settings.dtypes.float_type


class GPModelAdaptiveNoiseLengthscaleMultDim(Model):
    """
    A base class for adaptive GP (non-stationary lengthscale and signal variance)
    regression models with heteroscedastic noise,
    wherein, noise is represented by a latent GP N(.)
    """
    def __init__(self, X, Y, kern, nonstat, noisekern, name='adaptive_noise_lengthscale_gp_multdim'):
        Model.__init__(self, name)
        self.kern_type = kern
        self.nonstat = nonstat
        self.noisekern = noisekern
        self.likelihood = GaussianHeteroNoise()
        if isinstance(X, np.ndarray):
            #: X is a data matrix; each row represents one instance
            X = DataHolder(X)
        if isinstance(Y, np.ndarray):
            #: Y is a data matrix; rows correspond to the rows in X, columns are treated independently
            Y = DataHolder(Y)
        self.likelihood._check_targets(Y.value)
        self.X, self.Y = X, Y
        self._session = None
    
    @AutoFlow((float_type, [None, None]))
    def predict_l(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_l(Xnew)

    @AutoFlow((float_type, [None, None]))
    def predict_n(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_n(Xnew)

    @AutoFlow((float_type, [None, None]))
    def predict(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_f(Xnew)

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def pred_cov(self, X1, X2):
        """
        Compute the posterior covariance matrix b/w X1 and X2.
        """
        return self.build_pred_cov_f(X1, X2)

    @AutoFlow((float_type, [None, None]), (tf.int32, []))
    def posterior_samples_n(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        mu, var = self.build_predict_f(Xnew, full_cov=True)
        jitter = tf.eye(tf.shape(mu)[0], dtype=float_type) * settings.numerics.jitter_level
        mu_n, var_n = self.build_predict_n(Xnew)
        mu_n = tf.square(tf.exp(mu_n))
        A = var[:, :] + jitter
        B = tf.multiply(mu_n, tf.eye(tf.shape(mu_n)[0], dtype=float_type))
        L = tf.cholesky(A + B)
        shape = tf.stack([tf.shape(L)[0], num_samples])
        V = tf.random_normal(shape, dtype=settings.dtypes.float_type)
        samples = mu[:, ] + tf.matmul(L, V)
        return tf.transpose(samples)
    
    @AutoFlow((float_type, [None, None]), (tf.int32, []))
    def posterior_samples(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        mu, var = self.build_predict_f(Xnew, full_cov=True)
        jitter = tf.eye(tf.shape(mu)[0], dtype=float_type) * settings.numerics.jitter_level
        L = tf.cholesky(var[:, :] + jitter)
        shape = tf.stack([tf.shape(L)[0], num_samples])
        V = tf.random_normal(shape, dtype=settings.dtypes.float_type)
        samples = mu[:, ] + tf.matmul(L, V)
        return tf.transpose(samples)
    
    
class GPModelAdaptiveLengthscaleMultDim(Model):
    """
    A base class for adaptive GP (non-stationary lengthscale and signal variance)
    regression models with heteroscedastic noise,
    wherein, noise is represented by a latent GP N(.)
    """
    def __init__(self, X, Y, kern, nonstat, mean_func, name='adaptive_lengthscale_gp_multdim'):
        Model.__init__(self, name)
        self.kern_type = kern
        self.nonstat = nonstat
        self.mean_func = mean_func
        self.likelihood = Gaussian()
        if isinstance(X, np.ndarray):
            #: X is a data matrix; each row represents one instance
            X = DataHolder(X)
        if isinstance(Y, np.ndarray):
            #: Y is a data matrix; rows correspond to the rows in X, columns are treated independently
            Y = DataHolder(Y)
        self.likelihood._check_targets(Y.value)
        self.X, self.Y = X, Y
        self._session = None
    
    @AutoFlow((float_type, [None, None]))
    def predict_l(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_l(Xnew)

    @AutoFlow((float_type, [None, None]))
    def predict(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_f(Xnew)

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def pred_cov(self, X1, X2):
        """
        Compute the posterior covariance matrix b/w X1 and X2.
        """
        return self.build_pred_cov_f(X1, X2)
    
    @AutoFlow((float_type, [None, None]), (tf.int32, []))
    def posterior_samples(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        mu, var = self.build_predict_f(Xnew, full_cov=True)
        jitter = tf.eye(tf.shape(mu)[0], dtype=float_type) * settings.numerics.jitter_level
        L = tf.cholesky(var[:, :] + jitter)
        shape = tf.stack([tf.shape(L)[0], num_samples])
        V = tf.random_normal(shape, dtype=settings.dtypes.float_type)
        samples = mu[:, ] + tf.matmul(L, V)
        return tf.transpose(samples)


# Need to define a ParamList for listing all the parameters associated with lengthscale gps for each dimension
# A list of parameters.
#This allows us to store parameters in a list whilst making them 'visible' to the gpflow machinery. The correct usage is
# >>> my_list = gpflow.param.ParamList([Param1, Param2])


class GPModelAdaptiveLengthscaleMultDimDev(Model):
    """
    A base class for adaptive GP (non-stationary lengthscale and signal variance)
    regression models with heteroscedastic noise,
    wherein, noise is represented by a latent GP N(.)
    """
    def __init__(self, X, Y, kerns_list, nonstat, mean_funcs_list, name='adaptive_lengthscale_gp_multdim'):
        
        Model.__init__(self, name)

        # Introducing Paramlist to define kernels and mean functions for lengthscale gps associated with each dimensions

        self.kerns_list = ParamList(kerns_list)
        self.nonstat = nonstat
        self.mean_funcs_list = ParamList(mean_funcs_list)
        self.likelihood = Gaussian()

        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]
        self.num_feat = X.shape[1]

        if isinstance(X, np.ndarray):
            #: X is a data matrix; each row represents one instance
            X = DataHolder(X)
        if isinstance(Y, np.ndarray):
            #: Y is a data matrix; rows correspond to the rows in X, columns are treated independently
            Y = DataHolder(Y)

        # Sanity checks : making sure we have defined kernels and mean functions for each lengthscale gp
        if len(self.kerns_list) != self.num_feat:
            raise ValueError('kernels defined for each lengthscale gps != number of features')
        if len(self.mean_funcs_list) != self.num_feat:
            raise ValueError('mean functions defined for each lengthscale gps != number of features')

        self.likelihood._check_targets(Y.value)
        self.X, self.Y = X, Y
        self._session = None
    
    @AutoFlow((float_type, [None, None]))
    def predict_l(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_l(Xnew)

    @AutoFlow((float_type, [None, None]))
    def predict(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_f(Xnew)

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def pred_cov(self, X1, X2):
        """
        Compute the posterior covariance matrix b/w X1 and X2.
        """
        return self.build_pred_cov_f(X1, X2)
    
    @AutoFlow((float_type, [None, None]), (tf.int32, []))
    def posterior_samples(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        mu, var = self.build_predict_f(Xnew, full_cov=True)
        jitter = tf.eye(tf.shape(mu)[0], dtype=float_type) * settings.numerics.jitter_level
        L = tf.cholesky(var[:, :] + jitter)
        shape = tf.stack([tf.shape(L)[0], num_samples])
        V = tf.random_normal(shape, dtype=settings.dtypes.float_type)
        samples = mu[:, ] + tf.matmul(L, V)
        return tf.transpose(samples)

class GPModelAdaptiveLengthscaleMultDimEllSSDev(Model):
    """
    A base class for adaptive GP (non-stationary lengthscale and signal variance)
    regression models with input-dependent signal strength and lengthscale.
    """
    def __init__(self, X, Y, kerns_ell_list, kerns_ss_list, mean_funcs_ell_list, mean_funcs_ss_list, nonstat, name='adaptive_lengthscale_gp_multdim_ell_ss'):
        
        Model.__init__(self, name)

        # Introducing Paramlist to define kernels and mean functions for lengthscale gps associated with each dimensions
        self.kerns_ell_list = ParamList(kerns_ell_list)
        self.kerns_ss_list = ParamList(kerns_ss_list)
        self.nonstat = nonstat
        self.mean_funcs_ell_list = ParamList(mean_funcs_ell_list)
        self.mean_funcs_ss_list = ParamList(mean_funcs_ss_list)
        self.likelihood = Gaussian()
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]
        self.num_feat = X.shape[1]
        
        if isinstance(X, np.ndarray):
            #: X is a data matrix; each row represents one instance
            X = DataHolder(X)
        if isinstance(Y, np.ndarray):
            #: Y is a data matrix; rows correspond to the rows in X, columns are treated independently
            Y = DataHolder(Y)

        # Sanity checks : making sure we have defined kernels and mean functions for each lengthscale gp
        if len(self.kerns_ell_list) != self.num_feat:
            raise ValueError('kernels defined for each lengthscale gps != number of features')
        if len(self.kerns_ss_list) != self.num_feat:
            raise ValueError('kernels defined for each signal-strength gps != number of features')
        if len(self.mean_funcs_ell_list) != self.num_feat:
            raise ValueError('mean functions defined for each lengthscale gps != number of features')
        if len(self.mean_funcs_ss_list) != self.num_feat:
            raise ValueError('mean functions defined for each signal-strength gps != number of features')

        self.likelihood._check_targets(Y.value)
        self.X, self.Y = X, Y
        self._session = None
    
    @AutoFlow((float_type, [None, None]))
    def predict_l(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_l(Xnew)

    @AutoFlow((float_type, [None, None]))
    def predict_s(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_s(Xnew)

    @AutoFlow((float_type, [None, None]))
    def predict(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_f(Xnew)

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def pred_cov(self, X1, X2):
        """
        Compute the posterior covariance matrix b/w ```X1``` and ```X2```.
        """
        return self.build_pred_cov_f(X1, X2)
    
    @AutoFlow((float_type, [None, None]), (tf.int32, []))
    def posterior_samples(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        ```Xnew```.
        """
        mu, var = self.build_predict_f(Xnew, full_cov=True)
        jitter = tf.eye(tf.shape(mu)[0], dtype=float_type) * settings.numerics.jitter_level
        L = tf.cholesky(var[:, :] + jitter)
        shape = tf.stack([tf.shape(L)[0], num_samples])
        V = tf.random_normal(shape, dtype=settings.dtypes.float_type)
        samples = mu[:, ] + tf.matmul(L, V)
        return tf.transpose(samples)