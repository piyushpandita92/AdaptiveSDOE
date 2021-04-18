"""
Comparison with US, EI, PI.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import gpflow
from scipy.optimize import minimize
import math
import GPy
from pyDOE import *
from _core import *
# from _gp_model import *
from _saving_log import *
from _gp_model_mcmc import *
from _sampler import *
import time
import sys
from copy import copy
from scipy.stats import multivariate_normal
from scipy.stats import norm
start_time = time.time()

__all__ = ['COMPSampler']

class COMPSampler(KLSampler):
	"""
	This class computes the sensitivity of a set of inputs
	by taking the posterior expectation of the var of the
	corresponding effect functions.
	"""

	def __init__(self, X, Y, x_hyp, noisy, bounds, qoi_func, 
				qoi_idx=1,
				obj_func=None,
				true_func=None,
				comp_method_idx=None,
				ego_kern=GPy.kern.RBF,
				ell_kern=gpflow.kernels.RBF,
				noise_kern=gpflow.kernels.RBF,
				jitter=1e-4,
				mcmc_from_scratch=False,
				mcmc_samples=1500,
				map_max_iter=5000,
				hmc_burn=1500,
				hmc_thin=2,
				hmc_epsilon=0.00005,
				hmc_lmax=160,
				num_quad_points=100,
				energy=0.95,
				nugget=1e-3,
				kld_tol=1e-2,
				func_name='ex1',
				quad_points=None,
				quad_points_weight=None,
				max_it=50,
				ego=True,
				lat_points='train',
				_acc_ratio_factor=0.65,
				**kwargs):
		"""
		:param X:		the inputs of the training data as an array.
		:param Y:		the outputs of the training data as an array.
		:param idx: 	set of indicies for which the
						effect function is needed.
		:param all:		if all lower level indicies are needed as well.
		"""
		assert X.ndim == 2
		self.X = X
		assert Y.ndim == 2
		self.Y = Y
		assert self.X.shape[0] == self.Y.shape[0]
		KLSampler.__init__(self, X, Y, x_hyp, noisy, bounds, qoi_func, qoi_idx=qoi_idx, 
			true_func=true_func, obj_func=obj_func, ego_kern=ego_kern,
				ell_kern=ell_kern,
				noise_kern=noise_kern,
				jitter=jitter,
				mcmc_from_scratch=mcmc_from_scratch,
				mcmc_samples=mcmc_samples,
				map_max_iter=map_max_iter,
				hmc_burn=hmc_burn,
				hmc_thin=hmc_thin,
				hmc_epsilon=hmc_epsilon,
				hmc_lmax=hmc_lmax,
				num_quad_points=num_quad_points,
				energy=energy,
				nugget=nugget,
				kld_tol=kld_tol,
				func_name=func_name,
				quad_points=quad_points,
				quad_points_weight=quad_points_weight,
				max_it=max_it,
				lat_points='train',
				_acc_ratio_factor=0.65, 
				**kwargs)
		self.comp_method_idx = comp_method_idx

	def get_us_best(self, X_design, num_samp_gp, num_samp=1000):
		"""
		Uncertainty sampling based selection of next design.
		"""
		sample_df = self.samples_df
		m = self.model
		sigma_us = []
		for i in range(self.mcmc_samples - num_samp_gp, self.mcmc_samples):
			m.set_parameter_dict(sample_df.iloc[i])
			idx = i - (self.mcmc_samples - num_samp_gp)
			sigma_us_m = m.posterior_samples(X_design, num_samp)
			sigma_us.append(sigma_us_m)
		sigma_us = np.vstack(sigma_us)
		sigma_us_mcmc = np.var(sigma_us, axis=0)
		return sigma_us_mcmc

	def get_ei_best(self, X_design, num_samp_gp):
		"""
		Expected Improvement based selection of next design.
		"""
		sample_df = self.samples_df
		m = self.model
		if self.qoi_idx == 3:
			y_star = max(self.Y)
		elif self.qoi_idx == 4:
			y_star = min(self.Y)
		ei_mcmc = []
		for i in range(self.mcmc_samples - num_samp_gp, self.mcmc_samples):
			m.set_parameter_dict(sample_df.iloc[i])
			mu, sigma = m.predict(X_design)
			if self.qoi_idx == 3:
				ei_m = ei_for_max(mu, np.diag(sigma)[:, None], y_star)
			elif self.qoi_idx == 4:
				ei_m = ei(mu, np.diag(sigma)[:, None], y_star)
			ei_mcmc.append(ei_m)
		ei_mcmc = np.hstack(ei_mcmc)
		ei_final_mcmc = np.mean(ei_mcmc, axis=1)
		return ei_final_mcmc

	def get_pi_best(self, X_design, num_samp_gp):
		"""
		Probability of Improvement based selection of next design.
		"""
		sample_df = self.samples_df
		m = self.model
		if self.qoi_idx == 3:
			y_star = max(self.Y)
		elif self.qoi_idx == 4:
			y_star = min(self.Y)
		poi_mcmc = []
		for i in range(self.mcmc_samples - num_samp_gp, self.mcmc_samples):
			m.set_parameter_dict(sample_df.iloc[i])
			mu, sigma = m.predict(X_design)
			if self.qoi_idx == 3:
				poi_m = poi_for_max(mu, np.diag(sigma)[:, None], y_star)
			elif self.qoi_idx == 4:
				poi_m = poi(mu, np.diag(sigma)[:, None], y_star)
			poi_mcmc.append(poi_m)
		poi_mcmc = np.hstack(poi_mcmc)
		poi_final_mcmc = np.mean(poi_mcmc, axis=1)
		return poi_final_mcmc

	def optimize(self, num_designs=1000, verbose=0, plots=0, num_samp_gp=20, num_samp=50, num_post_samp=1000):
		"""
		:param num_designs:						A discretized set of hypothetical designs
		:param plots:							Plotting the response surface, the IAF, and the lengthscales.
		:param ekld_lhs_fac:					Fraction of ekld iterations to be used for initial design
		:return:								Final set of inputs, and outputs, ...
		"""
		iaf_rel = np.zeros(self.max_it)
		iaf_all = np.ndarray((self.max_it, num_designs))
		mu_qoi = []
		sigma_qoi = []
		for it in xrange(self.max_it):
			print 'iteration no. ', it + 1, 'of ', self.max_it
			kld = np.zeros(num_designs)
			val_trunc_mcmc, vec_trunc_mcmc, W_h_mcmc, x_d_mcmc, phi_x_dx_mcmc = self.get_val_vec_mcmc(num_samp_gp=num_samp_gp)
			samp_qd = self.qoi_qd_mcmc(val_trunc_mcmc=val_trunc_mcmc, vec_trunc_mcmc=vec_trunc_mcmc, W_h_mcmc=W_h_mcmc, x_d_mcmc=x_d_mcmc, phi_x_dx_mcmc=phi_x_dx_mcmc, num_samp=num_samp, num_samp_gp=num_samp_gp)
			mu_qd, sigma_qd = self.get_qd_mcmc(samp_qd)
			mu_qoi.append(mu_qd)
			sigma_qoi.append(sigma_qd)
			print '>... current mean of the QoI', mu_qd
			print '>... current variance of the QoI', sigma_qd
			X_design = lhs(self.X.shape[1], num_designs, criterion='center')
			if self.comp_method_idx == 'us':
				iaf = self.get_us_best(X_design, num_samp_gp)
			elif self.comp_method_idx == 'ei':
				print '>... computing EI: '
				iaf = self.get_ei_best(X_design, num_samp_gp)
			elif self.comp_method_idx == 'pi':
				print '>... computing PI: '
				iaf = self.get_pi_best(X_design, num_samp_gp)
			x_best = X_design[np.argmax(iaf)]
			if verbose > 0:
				print '>... run the next experiment at design: ', x_best
			y_obs = self.obj_func(x_best)
			if verbose > 0:
				print '>... simulated the output at the selected design', y_obs
			if plots > 0:
				self.make_plots(it, iaf, X_design, x_best, y_obs, 
					plots=plots, 
					num_post_samp=num_post_samp, 
					num_samp_gp=num_samp_gp)
			self.update_XY(x_best, y_obs)
			iaf_all[it, :] = iaf
			iaf_rel[it] = max(iaf)
			if verbose > 0:
				print '>... reconstructing surrogate model(s)'
			if self.acceptance_ratio[-1] < self._acc_ratio_factor:
				self.model, self.samples_df, self.acceptance_ratio = self.make_model(mcmc_from_scratch=True)
			else:
				self.model, self.samples_df, self.acceptance_ratio = self.make_model(mcmc_from_scratch=self.mcmc_from_scratch)
			tol_ratio = (max(iaf) / max(iaf_rel))
			if tol_ratio < self.kld_tol:
				print '>... relative ekld below specified tolerance ... stopping optimization now.'
				break
			if it == self.max_it-1:
				val_trunc_mcmc, vec_trunc_mcmc, W_h_mcmc, x_d_mcmc, phi_x_dx_mcmc = self.get_val_vec_mcmc(num_samp_gp=num_samp_gp)
				samp_qd = self.qoi_qd_mcmc(val_trunc_mcmc=val_trunc_mcmc, vec_trunc_mcmc=vec_trunc_mcmc, W_h_mcmc=W_h_mcmc, x_d_mcmc=x_d_mcmc, phi_x_dx_mcmc=phi_x_dx_mcmc, num_samp=num_samp, num_samp_gp=num_samp_gp)
				mu_qd, sigma_qd = self.get_qd_mcmc(samp_qd)
				mu_qoi.append(mu_qd)
				sigma_qoi.append(sigma_qd)
		return self.X, self.Y, iaf_all, X_design, mu_qoi, sigma_qoi

	def make_plots(self, it, iaf, X_design, x_best, y_obs, plots=1, num_post_samp=1000, num_samp_gp=20):
		# matplotlib.use('PS')
		# matplotlib.use('Qt4Agg')
		# import seaborn as sns
		sns.set_style("white")
		sns.set_context("paper")
		n = self.X.shape[0]
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax2 = ax1.twinx()
		x_grid = self.quad_points
		if self.true_func:
			y_grid = np.array([self.true_func(x_grid[i]) for i in xrange(x_grid.shape[0])])
			ax1.plot(x_grid, y_grid, '-', c=sns.color_palette()[0], linewidth=4.0, label='true function')
		sample_df = self.samples_df
		m = self.model
		if self.noisy:
			y_pos_n = []
		if plots > 1:
			y_m_ell = []
			y_m_ss = []
		y_pos = []
		for i in range(self.mcmc_samples - num_samp_gp, self.mcmc_samples):
			m.set_parameter_dict(sample_df.iloc[i])
			y_pos.append(m.posterior_samples(x_grid, num_post_samp))
			if self.noisy:
				y_pos_n.append(m.posterior_samples_n(x_grid, num_post_samp))
			if plots > 1:
				y_m_ell.append(m.predict_l(x_grid)[0])
				y_m_ss.append(m.predict_s(x_grid)[0])						# Note makes sense for a 1D function only.
		y_pos = np.vstack(y_pos)
		y_m = np.percentile(y_pos, 50, axis=0)
		y_l = np.percentile(y_pos, 2.5, axis=0)
		y_u = np.percentile(y_pos, 97.5, axis=0)
		if self.noisy:	
			y_pos_n = np.vstack(y_pos_n)
			y_m_n = np.percentile(y_pos_n, 50, axis=0)
			y_l_n = np.percentile(y_pos_n, 2.5, axis=0)
			y_u_n = np.percentile(y_pos_n, 97.5, axis=0)
			ax1.fill_between(x_grid[:, 0], y_l_n, y_u_n, color=sns.color_palette()[1], alpha=0.25, zorder=3)
		ax1.plot(x_grid, y_m, '--', c=sns.color_palette()[1], linewidth=3.0, label='NSGP', zorder=3)
		ax1.fill_between(x_grid[:, 0], y_l, y_u, color=sns.color_palette()[1], alpha=0.5, zorder=3)
		if it == self.max_it-1:
			ax1.scatter(x_best, y_obs, marker='X', s=80, c='black', zorder=10)
			ax1.scatter(self.X[:, 0], self.Y[:, 0], marker='X', s=80, c='black', label='observed data', zorder=10)
		else:
			ax1.scatter(x_best, y_obs, marker='D', s=80, c=sns.color_palette()[3], label='latest experiment', zorder=10)
			ax1.scatter(self.X[:, 0], self.Y[:, 0], marker='X', s=80, c='black', label='observed data', zorder=10)
		if plots > 2:
			idx = np.argsort(X_design[:, ], axis=0)[:, 0]
			ekld = ax2.plot(X_design[idx[:]], iaf[idx[:]], linestyle='-.', linewidth=3.0, c=sns.color_palette()[2], label='IAF', zorder=5)
		ax1.set_xlabel('$x$', fontsize=16)
		ax2.set_ylabel('$G(x)$', fontsize=16)
		ax2.set_ylim(0, 1)
		lines, labels = ax1.get_legend_handles_labels()
		lines2, labels2 = ax2.get_legend_handles_labels()
		ax2.legend(lines + lines2, labels + labels2, loc=9, fontsize=12)
		plt.xticks(fontsize=16)
		ax1.tick_params(axis='both', which='both', labelsize=16)
		ax2.tick_params(axis='both', which='both', labelsize=16)
		ax2.spines['right'].set_color(sns.color_palette()[2])
		ax2.yaxis.label.set_color(sns.color_palette()[2])
		ax2.tick_params(axis='y', colors=sns.color_palette()[2])
		ax1.set_ylabel('$f(x)$', fontsize=16)
		ax1.set_xlim(self.bounds["a"], self.bounds["b"])
		plt.savefig(self.func_name + '_kld_' + str(it + 1).zfill(len(str(self.max_it))) + '.png', dpi=(300), figsize=(3.25, 3.25))
		plt.clf()
		if plots > 1:
			fig = plt.figure()
			ax1 = fig.add_subplot(111)
			[ax1.plot(x_grid, np.exp(y_m_ell[i][0]), linestyle='-.', linewidth=3.0, c=sns.color_palette()[2], zorder=5) for i in xrange(num_samp_gp)]
			ax1.plot([-1], [-1], linestyle='-.', linewidth=3.0, c=sns.color_palette()[2], zorder=5, label='ell GP')
			ax1.set_xlabel('$x$', fontsize=16)
			ax1.set_ylabel('$ell(x)$', fontsize=16)
			plt.xticks(fontsize=16)
			plt.yticks(fontsize=16)
			plt.legend(fontsize=12)
			ax1.set_xlim(self.bounds["a"], self.bounds["b"])
			ax1.set_ylim(0, )
			plt.savefig(self.func_name + '_ell_' + str(it + 1).zfill(len(str(self.max_it))) + '.png', dpi=(300), figsize=(3.25, 3.25))
			plt.clf()
			plt.clf()
			fig = plt.figure()
			ax1 = fig.add_subplot(111)
			[ax1.plot(x_grid, np.exp(y_m_ss[i][0]), linestyle='-.', linewidth=3.0, c=sns.color_palette()[2], zorder=5) for i in xrange(num_samp_gp)]
			ax1.plot([-1], [-1], linestyle='-.', linewidth=3.0, c=sns.color_palette()[2], zorder=5, label='ss GP')
			ax1.set_xlabel('$x$', fontsize=16)
			ax1.set_ylabel('$ss(x)$', fontsize=16)
			plt.xticks(fontsize=16)
			plt.yticks(fontsize=16)
			plt.legend(fontsize=12)
			ax1.set_xlim(self.bounds["a"], self.bounds["b"])
			ax1.set_ylim(min([min(np.exp(y_m_ss[i][0])) for i in xrange(num_samp_gp)]), max([max(np.exp(y_m_ss[i][0])) for i in xrange(num_samp_gp)]))
			plt.savefig(self.func_name + '_ss_' + str(it + 1).zfill(len(str(self.max_it))) + '.png', dpi=(300), figsize=(3.25, 3.25))
			plt.clf()