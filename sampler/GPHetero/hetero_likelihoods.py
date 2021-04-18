from __future__ import absolute_import
from gpflow import densities, transforms
import tensorflow as tf
import numpy as np
from gpflow.param import Param
from gpflow.param import AutoFlow
from gpflow._settings import settings
from gpflow.likelihoods import Likelihood

float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64


class GaussianHeteroNoise(Likelihood):
    def __init__(self):
        Likelihood.__init__(self)

    def logp(self, F, N, Y):
        Nexp = tf.exp(N)
        Nvar = tf.square(Nexp)
        return densities.gaussian(F, Y, Nvar)
    
    @AutoFlow((float_type, [None, None]),
              (float_type, [None, None]),
              (float_type, [None, None]))
    def compute_logp(self, F, N, Y):
        return self.logp(F, N, Y)

class Gaussian(Likelihood):
    def __init__(self):
        Likelihood.__init__(self)
        self.variance = Param(1.0, transforms.positive)

    def logp(self, F, Y):
        return densities.gaussian(F, Y, self.variance)

    def conditional_mean(self, F):
        return tf.identity(F)

    def conditional_variance(self, F):
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    def predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    def predict_density(self, Fmu, Fvar, Y):
        return densities.gaussian(Fmu, Y, Fvar + self.variance)

    def variational_expectations(self, Fmu, Fvar, Y):
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(self.variance) \
               - 0.5 * (tf.square(Y - Fmu) + Fvar) / self.variance
               
    @AutoFlow((float_type, [None, None]),
              (float_type, [None, None]))
    def compute_logp(self, F, Y):
        return self.logp(F,Y)
    

if __name__ == '__main__':
    f = np.array([1,2,3])[:,None]
    n = np.array([0.1,0.2,0.3])[:,None]
    y = np.array([1,2,3])[:,None]
    
    L = GaussianHeteroNoise()
    L.compute_logp(f,n,y)
    
    P = Gaussian()
    
