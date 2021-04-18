import numpy as np
from copy import copy
import tensorflow as tf
from .hetero_model import GPModelAdaptiveNoiseLengthscaleMultDim, GPModelAdaptiveLengthscaleMultDim, GPModelAdaptiveLengthscaleMultDimDev, GPModelAdaptiveLengthscaleMultDimEllSSDev
import gpflow
from gpflow.param import Param, DataHolder
from .hetero_conditionals import conditional, nonstat_conditional_multdim, conditional_cov
from .hetero_conditionals_full import conditional_full, nonstat_conditional_multdim_full, conditional_cov_full
from .hetero_kernels import NonStationaryLengthscaleRBF, NonStationaryRBF
from gpflow.priors import Gaussian
from gpflow.mean_functions import Zero
from gpflow._settings import settings
float_type = settings.dtypes.float_type


class GPMCAdaptiveNoiseLengthscaleMultDim(GPModelAdaptiveNoiseLengthscaleMultDim):
    """ 
    X is a data matrix, size N x D
    Y is a data matrix, size N x 1
    kern1
    kern1: covariance function associated with adaptive lengthscale whose log is represented using GP L(.)
    This is a vanilla implementation of an adaptive GP (non-stationary lengthscale) 
    with a Gaussian likelihood
    
    v1 ~ N(0, I)
    l = L1v1 
    with
    L1 L1^T = K1 and 
    
    v2 ~ N(0, I)
    f = NonStatLv2
    with
    NonStatL NonStatL^T = NonStatK
    """
    def __init__(self, X, Y, kern, nonstat, noisekern, num_latent=None): 
        self.num_data = X.shape[0]
        self.num_latent = num_latent or Y.shape[1]
        self.num_feat = X.shape[1]
        # Standard normal dist for num_feat L(.) GP
        self.kerns = {}
        X = DataHolder(X, on_shape_change='recompile')
        Y = DataHolder(Y, on_shape_change='recompile')
        GPModelAdaptiveNoiseLengthscaleMultDim.__init__(self, X, Y, kern, nonstat, noisekern)
        for i in xrange(self.num_feat):
            self.kerns["ell"+str(i)] = self.kern_type
        self.V = Param(np.zeros((self.num_data, self.num_feat)))        # Lengthscales
        self.V.prior = Gaussian(0., 1.)
        self.V3 = Param(np.zeros((self.num_data, self.num_latent)))     # Noise variance
        self.V3.prior = Gaussian(0., 1.)
        self.V4 = Param(np.zeros((self.num_data, self.num_latent)))     # Signal strength
        self.V4.prior = Gaussian(0., 1.)
    
    def compile(self, session=None, graph=None, optimizer=None):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add parameters appropriately.

        This is necessary because the shape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X.shape[0]:
            self.num_data = self.X.shape[0]
            self.num_feat = self.X.shape[1]
            self.V = Param(np.zeros((self.num_data, self.num_feat)))
            self.V.prior = Gaussian(0., 1.)
            self.V3 = Param(np.zeros((self.num_data, self.num_latent)))
            self.V3.prior = Gaussian(0., 1.)
            self.V4 = Param(np.zeros((self.num_data, self.num_latent)))
            self.V4.prior = Gaussian(0., 1.)
        return super(GPMCAdaptiveNoiseLengthscaleMultDim, self).compile(session=session,
                                         graph=graph,
                                         optimizer=optimizer)

    def build_likelihood(self):
        """
        Construct a tf function to compute the likelihood of an adaptive GP
        model (non-stationary lengthscale whose 
        log is represented using latent GP L(.)).
        \log p(Y, V1, V2, V3, V4| theta).
        """
        # noise likelihood
        K3 = self.noisekern.K(self.X)
        L3 = tf.cholesky(K3 + tf.eye(tf.shape(self.X)[0], dtype=float_type) * settings.numerics.jitter_level)
        N = tf.matmul(L3, self.V3)
        # ell likelihood
        K_X_X = tf.ones(shape=[self.num_data, self.num_data], dtype=float_type)
        Xi_s = tf.split(self.X, num_or_size_splits = self.num_feat, axis = 1)
        Vi_s = tf.split(self.V, num_or_size_splits = self.num_feat, axis = 1)
        for i in xrange(self.num_feat):
            X_i = Xi_s[i]
            V_i = Vi_s[i]
            K_i = self.kerns["ell" + str(i)].K(X_i)
            L_i = tf.cholesky(K_i + tf.eye(tf.shape(X_i)[0], dtype=float_type) * 1e-4)
            Ls_i = tf.matmul(L_i, V_i)
            Ls_i_exp = tf.exp(Ls_i)
            K_X_X = tf.multiply(K_X_X, self.nonstat.K(X_i, Ls_i_exp, X_i, Ls_i_exp))
        K_X_X = self.nonstat.signal_variance * K_X_X
        Lnonstat = tf.cholesky(K_X_X + tf.eye(tf.shape(self.X)[0], dtype=float_type) * 1e-4)
        F = tf.matmul(Lnonstat, self.V4)
        return tf.reduce_sum(self.likelihood.logp(F, N, self.Y))

    def build_predict_l(self, Xnew, full_cov=False):
        """
        Predict latent lengthscale GP L(.) at new points Xnew
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(L* | (L=L1V1) )

        where L* are points on the GP at Xnew, N=L1V1 are points on the GP at X.

        """
        mu = []
        var = []
        Xi_s = tf.split(self.X, num_or_size_splits=self.num_feat, axis=1)
        Xnew_s = tf.split(Xnew, num_or_size_splits=self.num_feat, axis=1)
        Vi_s = tf.split(self.V, num_or_size_splits=self.num_feat, axis=1)
        for i in xrange(self.num_feat):
            X_i = Xi_s[i]
            Xnew_i = Xnew_s[i]
            V_i = Vi_s[i]
            mu_i, var_i = conditional(Xnew_i, X_i, self.kerns["ell"+str(i)], V_i,
                full_cov=full_cov, q_sqrt=None, whiten=True)
            mu.append(mu_i)
            var.append(var_i)
        return mu, var

    def build_predict_n(self, Xnew, full_cov=False):
        """
        Predict latent noise GP n(.) at Xnew.
        """
        mu, var = conditional(Xnew, self.X, self.noisekern, self.V3,
                              full_cov=full_cov,
                              q_sqrt=None, whiten=True)
        return mu, var

    def build_pred_cov_f(self, X1, X2):
        """
        Posterior covariance b/w X1 and X2.
        params X1:      nxd 
        params X2:      mxd.
        returns:    ```cov```, nxm.
        """
        mu_ell_X, var_ell_X = self.build_predict_l(self.X)
        mu_ell_X1, var_ell_X1 = self.build_predict_l(X1)
        mu_ell_X2, var_ell_X2 = self.build_predict_l(X2)
        cov = conditional_cov(X1, X2, self.X, mu_ell_X1, mu_ell_X2, mu_ell_X, self.nonstat, self.V4)
        return cov

    def build_predict_f(self, Xnew, full_cov=True):
        """
        Predict GP F(.) at new points Xnew
        Xnew is a data matrix, point at which we want to predict.
        This method computes
        p(F* | (L=LnonstatV2) )
        where L* are points on the GP at Xnew, N=L1V1 are points on the GP at X.
        """
        mu_ell_X, var_ell_X = self.build_predict_l(self.X)
        mu_ell_Xnew, var_ell_Xnew = self.build_predict_l(Xnew)
        mu, var = nonstat_conditional_multdim(Xnew, self.X, mu_ell_Xnew, mu_ell_X,  
            self.nonstat, self.V4, full_cov)
        return mu, var

class GPMCAdaptiveLengthscaleMultDim(GPModelAdaptiveLengthscaleMultDim):
    """ 
    X is a data matrix, size N x D
    Y is a data matrix, size N x 1
    kern1
    kern1: covariance function associated with adaptive lengthscale whose log is represented using GP L(.)
    This is a vanilla implementation of an adaptive GP (non-stationary lengthscale) 
    with a Gaussian likelihood
    
    v1 ~ N(0, I)
    l = L1v1 
    with
    L1 L1^T = K1 and 
    
    v2 ~ N(0, I)
    f = NonStatLv2
    with
    NonStatL NonStatL^T = NonStatK
    """
    def __init__(self, X, Y, kern, nonstat, mean_func, num_latent=None): 
        self.num_data = X.shape[0]
        self.num_latent = num_latent or Y.shape[1]
        self.num_feat = X.shape[1]
        # self.mean_func = mean_func
        # Standard normal dist for num_feat L(.) GP
        self.kerns = {}
        self.mean_funcs = {}
        X = DataHolder(X, on_shape_change='recompile')
        Y = DataHolder(Y, on_shape_change='recompile')
        GPModelAdaptiveLengthscaleMultDim.__init__(self, X, Y, kern, nonstat, mean_func)
        for i in xrange(self.num_feat):
            self.kerns["ell"+str(i)] = self.kern_type
            #self.kerns["ell"+str(i)] = gpflow.kernels.RBF(input_dim = 1)
            self.mean_funcs["ell"+str(i)] = self.mean_func
        self.V = Param(np.zeros((self.num_data, self.num_feat)))        # Lengthscales
        self.V.prior = Gaussian(0., 1.)
        self.V4 = Param(np.zeros((self.num_data, self.num_latent)))     # Signal strength
        self.V4.prior = Gaussian(0., 1.)
    
    def compile(self, session=None, graph=None, optimizer=None):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add parameters appropriately.

        This is necessary because the shape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X.shape[0]:
            self.num_data = self.X.shape[0]
            self.num_feat = self.X.shape[1]
            self.V = Param(np.zeros((self.num_data, self.num_feat)))
            self.V.prior = Gaussian(0., 1.)
            self.V4 = Param(np.zeros((self.num_data, self.num_latent)))
            self.V4.prior = Gaussian(0., 1.)
        return super(GPMCAdaptiveLengthscaleMultDim, self).compile(session=session,
                                         graph=graph,
                                         optimizer=optimizer)

    def build_likelihood(self):
        """
        Construct a tf function to compute the likelihood of an adaptive GP
        model (non-stationary lengthscale whose 
        log is represented using latent GP L(.)).
        \log p(Y, V1, V2, V3, V4| theta).
        """
        K_X_X = tf.ones(shape=[self.num_data, self.num_data], dtype=float_type)
        Xi_s = tf.split(self.X, num_or_size_splits = self.num_feat, axis = 1)
        Vi_s = tf.split(self.V, num_or_size_splits = self.num_feat, axis = 1)
        for i in xrange(self.num_feat):
            X_i = Xi_s[i]
            V_i = Vi_s[i]
            K_i = self.kerns["ell" + str(i)].K(X_i)
            L_i = tf.cholesky(K_i + tf.eye(tf.shape(X_i)[0], dtype=float_type) * 1e-4)
            Ls_i = tf.matmul(L_i, V_i) + self.mean_funcs["ell" + str(i)](X_i)
            Ls_i_exp = tf.exp(Ls_i)
            K_X_X = tf.multiply(K_X_X, self.nonstat.K(X_i, Ls_i_exp, X_i, Ls_i_exp))
        K_X_X = self.nonstat.signal_variance * K_X_X
        Lnonstat = tf.cholesky(K_X_X + tf.eye(tf.shape(self.X)[0], dtype=float_type) * 1e-4)
        F = tf.matmul(Lnonstat, self.V4)
        return tf.reduce_sum(self.likelihood.logp(F, self.Y))

    def build_predict_l(self, Xnew, full_cov=False):
        """
        Predict latent lengthscale GP L(.) at new points Xnew
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(L* | (L=L1V1) )

        where L* are points on the GP at Xnew, N=L1V1 are points on the GP at X.

        """
        mu = []
        var = []
        Xi_s = tf.split(self.X, num_or_size_splits=self.num_feat, axis=1)
        Xnew_s = tf.split(Xnew, num_or_size_splits=self.num_feat, axis=1)
        Vi_s = tf.split(self.V, num_or_size_splits=self.num_feat, axis=1)
        for i in xrange(self.num_feat):
            X_i = Xi_s[i]
            Xnew_i = Xnew_s[i]
            V_i = Vi_s[i]
            mu_i, var_i = conditional(Xnew_i, X_i, self.kerns["ell"+str(i)], V_i,
                full_cov=full_cov, q_sqrt=None, whiten=True)
            mu.append(mu_i + self.mean_funcs["ell" + str(i)](Xnew_i))
            var.append(var_i)
        return mu, var

    def build_pred_cov_f(self, X1, X2):
        """
        Posterior covariance b/w X1 and X2.
        params X1:      nxd 
        params X2:      mxd.
        returns:    ```cov```, nxm.
        """
        mu_ell_X, var_ell_X = self.build_predict_l(self.X)
        mu_ell_X1, var_ell_X1 = self.build_predict_l(X1)
        mu_ell_X2, var_ell_X2 = self.build_predict_l(X2)
        cov = conditional_cov(X1, X2, self.X, mu_ell_X1, mu_ell_X2, mu_ell_X, self.nonstat, self.V4)
        return cov

    def build_predict_f(self, Xnew, full_cov=True):
        """
        Predict GP F(.) at new points Xnew
        Xnew is a data matrix, point at which we want to predict.
        This method computes
        p(F* | (L=LnonstatV2) )
        where L* are points on the GP at Xnew, N=L1V1 are points on the GP at X.
        """
        mu_ell_X, var_ell_X = self.build_predict_l(self.X)
        mu_ell_Xnew, var_ell_Xnew = self.build_predict_l(Xnew)
        mu, var = nonstat_conditional_multdim(Xnew, self.X, mu_ell_Xnew, mu_ell_X,  
            self.nonstat, self.V4, full_cov)
        return mu, var

class GPMCAdaptiveLengthscaleMultDimDev(GPModelAdaptiveLengthscaleMultDimDev):
    """ 
    X is a data matrix, size N x D
    Y is a data matrix, size N x 1
    kern1
    kern1: covariance function associated with adaptive lengthscale whose log is represented using GP L(.)
    This is a vanilla implementation of an adaptive GP (non-stationary lengthscale) 
    with a Gaussian likelihood
    
    v1 ~ N(0, I)
    l = L1v1 
    with
    L1 L1^T = K1 and 
    
    v2 ~ N(0, I)
    f = NonStatLv2
    with
    NonStatL NonStatL^T = NonStatK
    """
    def __init__(self, X, Y, kerns_list, nonstat, mean_funcs_list): 
        X = DataHolder(X, on_shape_change='recompile')
        Y = DataHolder(Y, on_shape_change='recompile')
        GPModelAdaptiveLengthscaleMultDimDev.__init__(self, X, Y, kerns_list, nonstat, mean_funcs_list)
        
        self.V = Param(np.zeros((self.num_data, self.num_feat)))        # Lengthscales
        self.V.prior = Gaussian(0., 1.)
        self.V4 = Param(np.zeros((self.num_data, self.num_latent)))     # Signal strength
        self.V4.prior = Gaussian(0., 1.)
    
    def compile(self, session=None, graph=None, optimizer=None):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add parameters appropriately.

        This is necessary because the shape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X.shape[0]:
            self.num_data = self.X.shape[0]
            self.V = Param(np.zeros((self.num_data, self.num_feat)))
            self.V.prior = Gaussian(0., 1.)
            self.V4 = Param(np.zeros((self.num_data, self.num_latent)))
            self.V4.prior = Gaussian(0., 1.)
        return super(GPMCAdaptiveLengthscaleMultDimDev, self).compile(session=session,
                                         graph=graph,
                                         optimizer=optimizer)

    def build_likelihood(self):
        """
        Construct a tf function to compute the likelihood of an adaptive GP
        model (non-stationary lengthscale whose 
        log is represented using latent GP L(.)).
        \log p(Y, V1, V2, V3, V4| theta).
        """
        K_X_X = tf.ones(shape=[self.num_data, self.num_data], dtype=float_type)
        Xi_s = tf.split(self.X, num_or_size_splits = self.num_feat, axis = 1)
        Vi_s = tf.split(self.V, num_or_size_splits = self.num_feat, axis = 1)
        for i in xrange(self.num_feat):
            X_i = Xi_s[i]
            V_i = Vi_s[i]
            K_i = self.kerns_list[i].K(X_i)
            L_i = tf.cholesky(K_i + tf.eye(tf.shape(X_i)[0], dtype=float_type) * 1e-4)
            Ls_i = tf.matmul(L_i, V_i) + self.mean_funcs_list[i](X_i)
            Ls_i_exp = tf.exp(Ls_i)
            K_X_X = tf.multiply(K_X_X, self.nonstat.K(X_i, Ls_i_exp, X_i, Ls_i_exp))
        K_X_X = self.nonstat.signal_variance * K_X_X
        Lnonstat = tf.cholesky(K_X_X + tf.eye(tf.shape(self.X)[0], dtype=float_type) * 1e-4)
        F = tf.matmul(Lnonstat, self.V4)
        return tf.reduce_sum(self.likelihood.logp(F, self.Y))

    def build_predict_l(self, Xnew, full_cov=False):
        """
        Predict latent lengthscale GP L(.) at new points Xnew
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(L* | (L=L1V1) )

        where L* are points on the GP at Xnew, N=L1V1 are points on the GP at X.

        """
        mu = []
        var = []
        Xi_s = tf.split(self.X, num_or_size_splits=self.num_feat, axis=1)
        Xnew_s = tf.split(Xnew, num_or_size_splits=self.num_feat, axis=1)
        Vi_s = tf.split(self.V, num_or_size_splits=self.num_feat, axis=1)
        for i in xrange(self.num_feat):
            X_i = Xi_s[i]
            Xnew_i = Xnew_s[i]
            V_i = Vi_s[i]
            mu_i, var_i = conditional(Xnew_i, X_i, self.kerns_list[i], V_i,
                full_cov=full_cov, q_sqrt=None, whiten=True)
            mu.append(mu_i + self.mean_funcs_list[i](Xnew_i))
            var.append(var_i)
        return mu, var

    def build_pred_cov_f(self, X1, X2):
        """
        Posterior covariance b/w X1 and X2.
        params X1:      nxd 
        params X2:      mxd.
        returns:    ```cov```, nxm.
        """
        mu_ell_X, var_ell_X = self.build_predict_l(self.X)
        mu_ell_X1, var_ell_X1 = self.build_predict_l(X1)
        mu_ell_X2, var_ell_X2 = self.build_predict_l(X2)
        cov = conditional_cov(X1, X2, self.X, mu_ell_X1, mu_ell_X2, mu_ell_X, self.nonstat, self.V4)
        return cov

    def build_predict_f(self, Xnew, full_cov=True):
        """
        Predict GP F(.) at new points Xnew
        Xnew is a data matrix, point at which we want to predict.
        This method computes
        p(F* | (L=LnonstatV2) )
        where L* are points on the GP at Xnew, N=L1V1 are points on the GP at X.
        """
        mu_ell_X, var_ell_X = self.build_predict_l(self.X)
        mu_ell_Xnew, var_ell_Xnew = self.build_predict_l(Xnew)
        mu, var = nonstat_conditional_multdim(Xnew, self.X, mu_ell_Xnew, mu_ell_X,  
            self.nonstat, self.V4, full_cov)
        return mu, var

class GPMCAdaptiveLengthscaleMultDimEllSSDev(GPModelAdaptiveLengthscaleMultDimEllSSDev):
    """ 
    X is a data matrix, size N x D
    Y is a data matrix, size N x 1
    kern1
    kern1: covariance function associated with adaptive lengthscale whose log is represented using GP L(.)
    This is a vanilla implementation of an adaptive GP (non-stationary lengthscale) 
    with a Gaussian likelihood
    
    v1 ~ N(0, I)
    l = L1v1 
    with
    L1 L1^T = K1 and 
    
    v2 ~ N(0, I)
    f = NonStatLv2
    with
    NonStatL NonStatL^T = NonStatK
    """
    def __init__(self, X, Y, kerns_ell_list, kerns_ss_list, mean_funcs_ell_list, mean_funcs_ss_list, nonstat): 
        X = DataHolder(X, on_shape_change='recompile')
        Y = DataHolder(Y, on_shape_change='recompile')
        GPModelAdaptiveLengthscaleMultDimEllSSDev.__init__(self, X, Y, kerns_ell_list, kerns_ss_list, mean_funcs_ell_list, mean_funcs_ss_list, nonstat)
        
        self.V_ell = Param(np.zeros((self.num_data, self.num_feat)))        # Lengthscales
        self.V_ell.prior = Gaussian(0., 1.)
        self.V_ss = Param(np.zeros((self.num_data, self.num_feat)))         # Signal-strength
        self.V_ss.prior = Gaussian(0., 1.)
        self.V = Param(np.zeros((self.num_data, self.num_latent)))          # Latent function
        self.V.prior = Gaussian(0., 1.)
    
    def compile(self, session=None, graph=None, optimizer=None):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add parameters appropriately.

        This is necessary because the shape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X.shape[0]:
            self.num_data = self.X.shape[0]
            self.V_ell = Param(np.zeros((self.num_data, self.num_feat)))        # Lengthscales
            self.V_ell.prior = Gaussian(0., 1.)
            self.V_ss = Param(np.zeros((self.num_data, self.num_feat)))         # Signal-strength
            self.V_ss.prior = Gaussian(0., 1.)
            self.V = Param(np.zeros((self.num_data, self.num_latent)))          # Latent function
            self.V.prior = Gaussian(0., 1.)
        return super(GPMCAdaptiveLengthscaleMultDimEllSSDev, self).compile(session=session,
                                         graph=graph,
                                         optimizer=optimizer)

    def build_likelihood(self):
        """
        Construct a tf function to compute the likelihood of an adaptive GP
        model (non-stationary lengthscale whose 
        log is represented using latent GP L(.)).
        \log p(Y, V1, V2, V3, V4| theta).
        """
        K_X_X = tf.ones(shape=[self.num_data, self.num_data], dtype=float_type)
        Xi_s = tf.split(self.X, num_or_size_splits = self.num_feat, axis = 1)
        Vi_ell_s = tf.split(self.V_ell, num_or_size_splits = self.num_feat, axis = 1)
        Vi_ss_s = tf.split(self.V_ss, num_or_size_splits = self.num_feat, axis = 1)
        for i in xrange(self.num_feat):
            X_i = Xi_s[i]
            V_ell_i = Vi_ell_s[i]
            V_ss_i = Vi_ss_s[i]
            K_ell_i = self.kerns_ell_list[i].K(X_i)
            L_ell_i = tf.cholesky(K_ell_i + tf.eye(tf.shape(X_i)[0], dtype=float_type) * 1e-4)
            Ls_ell_i = tf.matmul(L_ell_i, V_ell_i) + self.mean_funcs_ell_list[i](X_i)
            Ls_i_exp = tf.exp(Ls_ell_i)
            K_ss_i = self.kerns_ss_list[i].K(X_i)
            L_ss_i = tf.cholesky(K_ss_i + tf.eye(tf.shape(X_i)[0], dtype=float_type) * 1e-4)
            Ls_ss_i = tf.matmul(L_ss_i, V_ss_i) + self.mean_funcs_ss_list[i](X_i)
            Ss_i_exp = tf.exp(Ls_ss_i)
            K_X_X = tf.multiply(K_X_X, self.nonstat.K(X_i, Ls_i_exp, Ss_i_exp, X_i, Ls_i_exp, Ss_i_exp))
        Lnonstat = tf.cholesky(K_X_X + tf.eye(tf.shape(self.X)[0], dtype=float_type) * 1e-8)
        F = tf.matmul(Lnonstat, self.V)
        return tf.reduce_sum(self.likelihood.logp(F, self.Y))

    def build_predict_l(self, Xnew, full_cov=False):
        """
        Predict latent lengthscale GP L(.) at new points ```Xnew```.
        Xnew is a data matrix, point at which we want to predict.
        This method computes:   p(L* | (L=L1V1)).
        where L* are points on the GP at Xnew, N=L1V1 are points on the GP at ```X```.
        """
        mu = []
        var = []
        Xi_s = tf.split(self.X, num_or_size_splits=self.num_feat, axis=1)
        Xnew_s = tf.split(Xnew, num_or_size_splits=self.num_feat, axis=1)
        Vi_s = tf.split(self.V_ell, num_or_size_splits=self.num_feat, axis=1)
        for i in xrange(self.num_feat):
            X_i = Xi_s[i]
            Xnew_i = Xnew_s[i]
            V_i = Vi_s[i]
            mu_i, var_i = conditional_full(Xnew_i, X_i, self.kerns_ell_list[i], V_i,
                full_cov=full_cov, q_sqrt=None, whiten=True)
            mu.append(mu_i + self.mean_funcs_ell_list[i](Xnew_i))
            var.append(var_i)
        return mu, var

    def build_predict_s(self, Xnew, full_cov=False):
        """
        Predict latent Signal-strength GP L(.) at new points ```Xnew```.
        ```Xnew``` is a data matrix, point at which we want to predict.
        This method computes:   p(L* | (L=L1V1)).
        where L* are points on the GP at Xnew, N=L1V1 are points on the GP at X.
        """
        mu = []
        var = []
        Xi_s = tf.split(self.X, num_or_size_splits=self.num_feat, axis=1)
        Xnew_s = tf.split(Xnew, num_or_size_splits=self.num_feat, axis=1)
        Vi_s = tf.split(self.V_ss, num_or_size_splits=self.num_feat, axis=1)
        for i in xrange(self.num_feat):
            X_i = Xi_s[i]
            Xnew_i = Xnew_s[i]
            V_i = Vi_s[i]
            mu_i, var_i = conditional_full(Xnew_i, X_i, self.kerns_ss_list[i], V_i,
                full_cov=full_cov, q_sqrt=None, whiten=True)
            mu.append(mu_i + self.mean_funcs_ss_list[i](Xnew_i))
            var.append(var_i)
        return mu, var

    def build_pred_cov_f(self, X1, X2):
        """
        Posterior covariance b/w X1 and X2.
        params X1:      nxd 
        params X2:      mxd.
        returns:        ```cov```, nxm.
        """
        mu_ell_X, var_ell_X = self.build_predict_l(self.X)
        mu_ss_X, var_ss_X = self.build_predict_s(self.X)
        mu_ell_X1, var_ell_X1 = self.build_predict_l(X1)
        mu_ss_X1, var_ss_X1 = self.build_predict_s(X1)
        mu_ell_X2, var_ell_X2 = self.build_predict_l(X2)
        mu_ss_X2, var_ss_X2 = self.build_predict_s(X2)
        cov = conditional_cov_full(X1, X2, self.X, mu_ell_X1, mu_ell_X2, mu_ell_X, mu_ss_X1, mu_ss_X2, mu_ss_X, self.nonstat, self.V)
        return cov

    def build_predict_f(self, Xnew, full_cov=True):
        """
        Predict GP F(.) at new points Xnew
        Xnew is a data matrix, point at which we want to predict.
        This method computes
        p(F* | (L=LnonstatV2) )
        where L* are points on the GP at Xnew, N=L1V1 are points on the GP at X.
        """
        mu_ell_X, var_ell_X = self.build_predict_l(self.X)
        mu_ss_X, var_ss_X = self.build_predict_s(self.X)
        mu_ell_Xnew, var_ell_Xnew = self.build_predict_l(Xnew)
        mu_ss_Xnew, var_ss_Xnew = self.build_predict_s(Xnew)
        mu, var = nonstat_conditional_multdim_full(Xnew, self.X, mu_ell_Xnew, mu_ell_X,  mu_ss_Xnew, mu_ss_X, self.nonstat, self.V, full_cov)
        return mu, var