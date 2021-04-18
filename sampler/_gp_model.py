
"""
Returns a NSGP model object.
"""
import GPy
import tqdm
from _core import *
import itertools
import time
from pyDOE import *
import scipy
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.optimize import check_grad

__all__ = ['NSGPModel']

class NSGPModel(object):
	"""
	NSGP model object.
	"""
	def __init__(self, X, Y, X_m, X_val, Y_val, 
		lat_model_kern=GPy.kern.Matern32, 
		num_lat_points=5,
		l_params_bounds=None, 
		nugget=1e-3, 
		jitter=1e-4,
		num_designs_l_hyp=50,
		num_opt_restarts=20, 
		opt_bounds=None):
		assert X.ndim == 2
		self.X = X
		assert Y.ndim == 2
		self.Y = Y
		assert self.X.shape[0] == self.Y.shape[0]
		if X_val is not None:
			self.X_val = X_val
			self.Y_val = Y_val
		else:
			self.X_val = X
			self.Y_val = Y
		self.dim = self.X.shape[1]
		self.num_obj = self.Y.shape[1]
		self.X_m = X_m
		self.num_lat_points = self.X_m.shape[0] 
		if l_params_bounds is not None:
			self.l_params_bounds = l_params_bounds
		else:
			self.l_params_bounds = [(1e-4, 1e-6), 	# noise variance of latent GP
									(0.0, 1.),	 	# signal strength of latent GP
									(0.0, 1.)]
		self.nugget = nugget
		self.jitter = jitter
		self.lat_model_kern = lat_model_kern
		self.num_opt_restarts = num_opt_restarts
		self.num_designs_l_hyp = num_designs_l_hyp
		self.opt_bounds = opt_bounds
		self._sigma_l = 1e-3
		self._ss_l = 1.
		self._ell_l = 0.3
		self._lengthscale_factor = 0.5
		self._nugget_factor = 0.5
		self._signalstrength_factor = 1.
		self.A_inv = None

	def kern_mat(self, xi, xj):
		"""
		Computes an ```nxn``` matrix whose elements are the RBF kernel based values for the two
		input arrays. This is the prior covariance.
		:param xi:		array of input(s)
		:param xj:		array of input(s)
		"""
		k = self.model[0].kern.K(xi, xj)
		return k

	def get_dist_mat(self, X_i, X_j=None):
		"""
		:param X_i: 	array of inputs
		:param X_j: 	array of an input
		"""
		if X_j is None:
			return scipy.spatial.distance.cdist(X_i, X_i) ** 2
		else:
			return scipy.spatial.distance.cdist(X_i, X_j) ** 2

	def make_model_l(self, l_lat=None, l_gp_params=None):
		"""
		A stationary GP model for the lengthscale process.
		:param l_lat:			latent-lengthscale(s) GP's training data
		:param sigma_l:	 		noise variance of the lengthscale GP
		:param ss_l:			signal-strength i.e. s, of the lengthscale GP
		:param ell_l:			lengthscale of the lengthscale GP 
		"""
		lat_m = GPy.models.GPRegression(self.X_m, np.atleast_2d(l_lat), self.lat_model_kern(input_dim=self.dim, ARD=True))
		lat_m.kern.lengthscale.fix(l_gp_params[2:], warning=False)
		lat_m.kern.variance.fix(l_gp_params[1] ** 2., warning=False)
		lat_m.likelihood.variance.fix(l_gp_params[0] ** 2., warning=False)
		return lat_m

	def make_model_n(self, n_lat=None, n_gp_params=None):
		"""
		A stationary GP model for the lengthscale process.
		:param l_lat:			latent-lengthscale(s) GP's training data
		:param sigma_l:	 		noise variance of the lengthscale GP
		:param ss_l:			signal-strength i.e. s, of the lengthscale GP
		:param ell_l:			lengthscale of the lengthscale GP 
		"""
		lat_m = GPy.models.GPRegression(self.X_m, np.atleast_2d(n_lat), self.lat_model_kern(input_dim=self.dim, ARD=True))
		lat_m.kern.lengthscale.fix(n_gp_params[2:], warning=False)
		lat_m.kern.variance.fix(n_gp_params[1] ** 2., warning=False)
		lat_m.likelihood.variance.fix(n_gp_params[0] ** 2., warning=False)
		return lat_m

	def grad_log_obj(self, params, l_gp_params, n_gp_params):
		"""
		Gradients of the objective function wrt. the parameters ```l_bar```, 
		and the hyper-parameters of the non-stationary GP (excluding the 
		hyper-parameters of the latent-lengthscale process).
		:param params:			parameters to be calibrated in the inner loop
		:param sigma_l:			noise variance of the lengthscale GP
		:param ss_l:			signal-strength of the lengthscale GP
		:param ell_l:			lengthscale of the lengthscale GP
		"""
		grads_l = []
		grads_sigma_f = []
		assert len(params) == 2 * self.num_lat_points + 1
		l_lat = params[:self.num_lat_points]
		n_lat = params[self.num_lat_points:2 * self.num_lat_points]
		ss_f = params[-1]
		self.model_l = self.make_model_l(l_lat=l_lat[:, None], l_gp_params=l_gp_params)
		self.model_n = self.make_model_n(n_lat=n_lat[:, None], n_gp_params=n_gp_params)
		l = np.exp(self.model_l.predict(self.X)[0])	
		n = np.exp(self.model_n.predict(self.X)[0])
		assert len(l) == len(n)
		p = l ** 2.
		p_r = np.matmul(p, np.ones((1, len(p))))
		p_c = np.matmul(np.ones((1, len(p))).T, p.T)
		P_r = np.matmul(p, np.ones((1, len(p)))) ** (1. * self.dim)
		P_c = np.matmul(np.ones((1, len(p))).T, p.T) ** (1. * self.dim)
		P_s = (p_r + p_c) ** (1. * self.dim)
		P_d = p_r + p_c
		s_X = self.get_dist_mat(self.X)
		E = np.exp(((-1. * s_X) / P_d))		# TODO: Make sure the denominator has ```P_d```. Comment: DONE
		K_x_x = (ss_f ** 2.) * (np.sqrt(2.) ** self.dim) * np.multiply(np.multiply(np.multiply((P_r ** (1 / 4.)), (P_c ** (1 / 4.))), (P_s ** (-1 / 2.))), E)
		K_x_x_l = self.model_l.kern.K(self.X_m, self.X_m)
		K_x_x_n = self.model_n.kern.K(self.X_m, self.X_m)
		A = K_x_x + np.multiply(n ** 2, np.eye(n.shape[0]))
		B = K_x_x_l  + (l_gp_params[0] ** 2.) * np.eye(K_x_x_l.shape[0])
		C = K_x_x_n  + (n_gp_params[0] ** 2.) * np.eye(K_x_x_n.shape[0])
		try:
			A_inv = np.linalg.inv(A)
			B_inv = np.linalg.inv(B)
			C_inv = np.linalg.inv(C)
		except:
			print ">... trying with a larger jitter"
			try:
				A_inv = np.linalg.inv(A + self.jitter * np.eye(A.shape[0]))
				B_inv = np.linalg.inv(B + self.jitter * np.eye(B.shape[0]))
				C_inv = np.linalg.inv(C + self.jitter * np.eye(C.shape[0]))
			except:
				import pdb
				pdb.set_trace()
		# A_inv = np.linalg.inv(A)
		# B_inv = np.linalg.inv(B)			# same as ```self.model_l.posterior.woodbury_inv```
		# grad_obj_sigma_f = (2. * sigma_f) * (-1. * np.matmul(np.matmul(self.Y.T, A_inv), np.matmul(A_inv, self.Y)) + np.trace(A_inv)) # gradient wrt the noise variacne of the GP 
		# The following gradient is different than the one in the paper
		grad_obj_ss_f = (2. / ss_f) * (-1. * np.matmul(np.matmul(np.matmul(self.Y.T, A_inv), K_x_x), np.matmul(A_inv, self.Y)) + np.trace(np.matmul(A_inv, K_x_x))) # gradient wrt the signal strength of the GP 
		# Now we code the gradients of the negative of the objective function wrt. the latent lengthscales
		#K_x_X = np.vstack([self.model_l.kern.K(np.atleast_2d(x), self.X_m) for x in self.X]) # nxm to be multiplied with B_inv
		K_x_X = self.model_l.kern.K(self.X, self.X_m)
		W = np.matmul(K_x_X, B_inv)
		M = np.matmul(K_x_X, C_inv)
		for j in xrange(len(l_lat)):
			# sigma_f_grad_fac = np.ones(len(l_lat))
			grad_pr =  2. *  np.multiply((W[:, j])[:, None], p_r)
			grad_pr_l_j = (self.dim * 2. / 4) *  np.multiply((W[:, j])[:, None], (P_r ** (1 / 4.)))
			grad_pc_l_j = grad_pr_l_j.T
			grad_ps_l_j = -(self.dim * 0.5) * np.multiply((P_d ** (-1 - self.dim / 2.)), (grad_pr + grad_pr.T))
			grad_E_l_j = 1. * np.multiply(np.multiply(np.exp(-1. * s_X / (P_d)), s_X / (P_d ** (2))), (grad_pr + grad_pr.T))
			grad_k_x_x_l_j = (ss_f ** 2.) * (np.sqrt(2.) ** self.dim) * (np.multiply(np.multiply(grad_pr_l_j, P_c ** (1 / 4.)), np.multiply(P_s ** (-1 / 2.), E)) + np.multiply(np.multiply(P_r ** (1 / 4.), grad_pc_l_j), np.multiply(P_s ** (-1 / 2.), E)) + np.multiply(np.multiply(P_r ** (1 / 4.), P_c ** (1 / 4.)), np.multiply(grad_ps_l_j, E)) + np.multiply(np.multiply(P_r ** (1 / 4.), P_c ** (1 / 4.)), np.multiply(P_s ** (-1 / 2.), grad_E_l_j)))
			grad_l_j = -1. * np.matmul(np.matmul(np.matmul(self.Y.T, A_inv), grad_k_x_x_l_j), np.matmul(A_inv, self.Y)) + np.trace(np.matmul(A_inv, grad_k_x_x_l_j)) 
			grads_l.append(grad_l_j[0, 0])
			# sigma_f_grad_fac[j] = 2
			grad_a_sigma_f_j = np.multiply(2. *  np.multiply((M[:, j])[:, None], n ** 2), np.eye(n.shape[0]))
			# grad_a_sigma_f_j = np.matmul(sigma_f_grad_fac[:, None], np.multiply(n ** 2, np.eye(n.shape[0])))
			grad_obj_sigma_f_j = -1 * np.matmul(np.matmul(np.matmul(self.Y.T, A_inv), grad_a_sigma_f_j), np.matmul(A_inv, self.Y)) + np.trace(np.matmul(A_inv, grad_a_sigma_f_j))
			grads_sigma_f.append(grad_obj_sigma_f_j[0, 0])
		# return 0.5 * np.hstack([grads_l, grad_obj_sigma_f[0, 0], grad_obj_ss_f[0, 0]])
		return 0.5 * np.hstack([grads_l, grads_sigma_f, grad_obj_ss_f[0, 0]])

	def log_obj_opt(self, params, l_gp_params, n_gp_params):
		"""
		The objective function to be optimized wrt. the values in the ```theta``` vector.
		:param params:		array of latent GP values for lengthscale, noise variance 
							and scalar value of signal-strength, in that order
		:param l_lat: 		array of supporting lengthscales at the observations
		:param sigma_f:		noise in the original process
		:param ss_f:		signal-strength of the original process
		:param sigma_l:		noise in the latent-lengthscale process
		:param ss_l:		signal-strength of the latent-lengthscale process
		:param ell_l:		lengthscale of the latent-lengthscale process
		"""
		assert len(params) == 2 * self.num_lat_points + 1
		l_lat = params[:self.num_lat_points]
		n_lat = params[self.num_lat_points:2 * self.num_lat_points]
		# sigma_f = params[-2]
		ss_f = params[-1]
		self.model_l = self.make_model_l(l_lat=l_lat[:, None], l_gp_params=l_gp_params)
		self.model_n = self.make_model_n(n_lat=n_lat[:, None], n_gp_params=n_gp_params)
		l = np.exp(self.model_l.predict(self.X)[0])	
		n = np.exp(self.model_n.predict(self.X)[0])
		K_x_x = self.cov_func_mat(self.X, self.X, params, l_gp_params)
		K_x_x_l = self.model_l.kern.K(self.X_m, self.X_m)
		K_x_x_n = self.model_n.kern.K(self.X_m, self.X_m)
		A = K_x_x + np.multiply(n ** 2, np.eye(n.shape[0]))
		B = K_x_x_l  + (l_gp_params[0] ** 2.) * np.eye(K_x_x_l.shape[0])
		C = K_x_x_n  + (n_gp_params[0] ** 2.) * np.eye(K_x_x_n.shape[0])
		try:
			A_inv = np.linalg.inv(A)
		except:
			print "trying with a larger jitter"
			try:
				A = A + self.jitter * np.eye(A.shape[0])
				A_inv = np.linalg.inv(A)
			except:
				import pdb
				pdb.set_trace()
		log_obj = 1.5 * (self.X.shape[0] * np.log(2. * np.pi)) + 0.5 * ((np.matmul(np.matmul(self.Y.T, A_inv), self.Y)) + np.log(np.linalg.det(A)) + np.log(np.linalg.det(B)) + np.log(np.linalg.det(C))) 
		return log_obj

	def cov_func_mat(self, Xi, Xj, params, l_gp_params):
		"""
		Covariance matrix between Xi and Xj.
		Note: Currently only for the isotropic lengthscale non-stationary GPs.
		:param xi: 		a vector input(s)
		:param xj: 		a vector input(s)
	 	:param ss_f:	signal strength of the process
	 	:param li:		lengthscale(s) at ```Xi```
	 	:param lj:		lengthscale(s) at ```Xj```
		"""
		assert len(params) == 2 * self.num_lat_points + 1
		l_lat = params[:self.num_lat_points]
		ss_f = params[-1]
		self.model_l = self.make_model_l(l_lat=l_lat[:, None], l_gp_params=l_gp_params )
		li = np.exp(self.model_l.predict(Xi)[0])
		lj = np.exp(self.model_l.predict(Xj)[0])
		p_r = np.repeat(li ** 2., Xj.shape[0], axis=1)
		p_c = np.repeat((lj ** 2.).T, Xi.shape[0], axis=0)
		P_r = p_r ** self.dim
		P_c = p_c ** self.dim
		P_d = p_r + p_c
		P_s = P_d ** self.dim
		E = np.exp(-1. * self.get_dist_mat(Xi, Xj) / P_d)
		K_Xi_Xj = (ss_f ** 2) * (np.sqrt(2.) ** self.dim) * np.multiply(np.multiply(np.multiply((P_r ** (1 / 4.)) , (P_c ** (1 / 4.))), (P_s ** (-1 / 2.))), E)
		return K_Xi_Xj

	def pred_cov(self, Xi, Xj=None, params=None, include_likelihood=True, l_gp_params=None, n_gp_params=None, A_inv=None):
		"""
		Computes the predictive covariance b/w ```Xi``` and ```Xj```.
		"""
		check_noise = False
		if Xj is None:
			Xj = Xi
			check_noise = True
		if params is None:
			params = self.get_params()['nsgp_params']
		if n_gp_params is None:
			n_gp_params = self.get_params()['n_gp_params']
		if l_gp_params is None:
			l_gp_params = self.get_params()['l_gp_params']
		l_lat = params[:self.num_lat_points]
		n_lat = params[self.num_lat_points:2 * self.num_lat_points]
		ss_f = params[-1]
		self.model_l = self.make_model_l(l_lat=l_lat[:, None],l_gp_params= l_gp_params)
		self.model_n = self.make_model_n(n_lat=n_lat[:, None], n_gp_params= n_gp_params)
		K_x_s_x_s = self.cov_func_mat(Xi, Xj, params, l_gp_params)
		K_Xi_X = self.cov_func_mat(Xi, self.X, params, l_gp_params)
		K_Xj_X = self.cov_func_mat(Xj, self.X, params, l_gp_params)
		if A_inv is None:
			K_x_x = self.cov_func_mat(self.X, self.X, params, l_gp_params)
			n = np.exp(self.model_n.predict(self.X)[0])
			A = K_x_x + np.multiply(n ** 2, np.eye(n.shape[0]))
			A_inv = np.linalg.inv(A)
		# var = K_x_s_x_s - np.matmul(K_x_X, np.matmul(A_inv, K_x_X.T))
		var = K_x_s_x_s - np.matmul(K_Xi_X, np.matmul(A_inv, K_Xj_X.T))
		if include_likelihood:
			if check_noise:
				n_pred = np.exp(self.model_n.predict(Xi)[0])
				return var + np.multiply(n_pred ** 2, np.eye(n_pred.shape[0]))
		else:
			return var

	def pred_mean_ns(self, X, params, full_cov=False, include_likelihood=True, l_gp_params=None, n_gp_params=None, A_inv=None):
		"""
		Computes the predictive mean at given input(s) ```X``` given the values of the 
		hyper-parameters.
		"""
		l_lat = params[:self.num_lat_points]
		n_lat = params[self.num_lat_points:2 * self.num_lat_points]
		ss_f = params[-1]
		self.model_l = self.make_model_l(l_lat=l_lat[:, None],l_gp_params= l_gp_params)
		self.model_n = self.make_model_n(n_lat=n_lat[:, None], n_gp_params= n_gp_params)
		K_x_x = self.cov_func_mat(self.X, self.X, params, l_gp_params)
		if A_inv is None:
			n = np.exp(self.model_n.predict(self.X)[0])
			A = K_x_x + np.multiply(n ** 2, np.eye(n.shape[0]))
			try:
				A_inv = np.linalg.inv(A)
			except:
				A_inv = np.linalg.inv(A + self.jitter * np.eye(A.shape[0]))
		K_x_X = self.cov_func_mat(X, self.X, params, l_gp_params)
		mu = np.matmul(K_x_X, np.matmul(A_inv, self.Y))
		var = self.pred_cov(X, params=params, include_likelihood=include_likelihood, l_gp_params= l_gp_params, n_gp_params= n_gp_params, A_inv=A_inv) 
		if full_cov:
			return (mu, var)
		else:
			return (mu, np.diagonal(var)[:, None])

	def predict(self, X, full_cov=False, include_likelihood=True):
		"""
		Predict the output at X. A faster method to ```pred_mean_ns``` as it uses the
		A_inv which has been saved off.
		"""
		model_params = self.get_params()['nsgp_params']
		l_gp_params = self.get_params()['l_gp_params']
		n_gp_params = self.get_params()['n_gp_params']
		preds = self.pred_mean_ns(X=X, params=model_params, full_cov=full_cov, include_likelihood=include_likelihood, l_gp_params=l_gp_params, n_gp_params=n_gp_params, A_inv=self.A_inv)
		return preds

	def posterior_samples(self, X, n_samp=1, full_cov=False, include_likelihood=True, A_inv=None):
		"""
		Sample the non-stationary GP at X.
		:params X:	an array of input(s).	
		:params n_samp: number of samples of the function.
		:params full_cov:	this computes the full covariance of the Xs.
		"""
		preds = self.predict(X=X, full_cov=full_cov, include_likelihood=include_likelihood)
		return np.random.multivariate_normal(preds[0][:, 0], preds[1], n_samp)

	def _get_val_error(self, params, l_gp_params, n_gp_params):
		"""
		Computes the mean squared error on the separate validation data.
		"""
		Y_val_pred = self.pred_mean_ns(X=self.X_val, params=params, l_gp_params=l_gp_params, n_gp_params=n_gp_params)[0]
		return np.sum((Y_val_pred - self.Y_val) ** 2.)

	def get_params(self):
		"""
		Get the calibrated hyper-parameters of the non-stationary GP.
		"""
		return {'nsgp_params':self.nsgp_params, 'l_gp_params':self.l_gp_params, 'n_gp_params':self.n_gp_params}

	def set_params(self, params):
		"""
		Sets the parameters of the surrogate GP
		"""
		self.nsgp_params = params['nsgp_params']
		self.l_gp_params = params['l_gp_params']
		self.n_gp_params = params['n_gp_params']

	def set_A_inv(self):
		"""
		saving off the A_inv after the model has been trained.
		"""
		params = self.get_params()
		K_x_x = self.cov_func_mat(self.X, self.X, params['nsgp_params'], params['l_gp_params'])
		n_lat = params['nsgp_params'][self.num_lat_points:2 * self.num_lat_points]
		self.model_n = self.make_model_n(n_lat=n_lat[:, None], n_gp_params= params['n_gp_params'])
		n = np.exp(self.model_n.predict(self.X)[0])
		A = K_x_x + np.multiply(n ** 2, np.eye(n.shape[0]))
		self.A_inv = np.linalg.inv(A)

	def get_A_inv(self):
		"""
		getter for the saved ```A_inv```.
		"""
		return self.A_inv

	def make_model(self):
		"""
		Calibrates the parameters of the surrogate non-stationary GP(s).
		"""
		m = self.Y.shape[1]
		surrogates = []
		for i in xrange(m):
			if isinstance(self.num_designs_l_hyp, int):
				params_l = lhs(2 + self.dim, self.num_designs_l_hyp)
				if self.l_params_bounds:
					b = np.array(self.l_params_bounds)
					params_l = b[:, 0] + (b[:, 1] - b[:, 0]) * params_l
				params_n = params_l.copy()
				err = np.zeros(params_l.shape[0])
				m_params = np.ndarray((params_l.shape[0], 2 * self.num_lat_points + 1))
			else:
				params_l = np.concatenate([[self._sigma_l], [self._ss_l], [self._ell_l] * self.dim])[None, :]
				m_params = np.ndarray((params_l.shape[0], self.num_lat_points + 2))
			for k in xrange(params_l.shape[0]):
				# print params_l[k, :]
				opt_res = scipy.optimize.minimize(fun=self.log_obj_opt, 
					x0= np.hstack([self._lengthscale_factor * np.random.rand(self.num_lat_points), self._nugget_factor * np.random.rand(self.num_lat_points), self._signalstrength_factor * np.random.rand(1)]), 
					method='L-BFGS-B', 
					jac=self.grad_log_obj, 
					bounds=np.concatenate([[self.opt_bounds['ell_f']] * (self.num_lat_points), [self.opt_bounds['sigma_f']] * (self.num_lat_points), [self.opt_bounds['ss_f']]]), 
					# args= (params_l[k, 0], params_l[k, 1], params_l[k, 2:]), 
					args=(params_l[k, :], params_n[k, :]),
					options={'maxiter':500})
				# print 'using L-BFGS-B', opt_res
				# print 'scipy_grad', scipy.optimize.approx_fprime(opt_res.x, self.log_obj_opt, 1e-6, *(params_l[k, :], params_n[k, :]))
				# print 'grad_analytic', self.grad_log_obj(params=opt_res.x, l_gp_params=params_l[k, :], n_gp_params=params_n[k, :])
				if isinstance(self.num_designs_l_hyp, int):
					m_params[k, ] = opt_res.x
					err[k] = self._get_val_error(m_params[k, ], l_gp_params=params_l[k, :], n_gp_params=params_n[k, :])
				else:
					m_params[k, ] = opt_res.x
			if isinstance(self.num_designs_l_hyp, int):
				m_best = {'nsgp_params':m_params[np.argmin(err), :], 'l_gp_params':params_l[np.argmin(err), :], 'n_gp_params':params_n[np.argmin(err), :]}
				# print 'validation error', err
			else:
				m_best = {'nsgp_params':m_params[0, :], 'l_gp_params':params_l[0, :], 'n_gp_params':params_n[np.argmin(err), :]}
			print 'parameters selected', m_best
			surrogates.append(m_best)
		self.set_params(params=m_best)
		self.set_A_inv()

	def get_model_l(self):
		l_lat = np.atleast_2d(self.get_params()['nsgp_params'][:self.num_lat_points]).T
		model_l = self.make_model_l(l_lat=l_lat, l_gp_params=self.get_params()['l_gp_params'])
		return model_l

	def get_model_n(self):
		n_lat = np.atleast_2d(self.get_params()['nsgp_params'][self.num_lat_points:2 * self.num_lat_points]).T
		model_n = self.make_model_n(n_lat=n_lat, n_gp_params=self.get_params()['n_gp_params'])
		return model_n