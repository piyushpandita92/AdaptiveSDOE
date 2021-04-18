import gpflow
import matplotlib
matplotlib.use('agg')
import numpy as np
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import math
import os
import sys
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pdb
import GPy
import time
import itertools
import pickle
from scipy.stats import norm
from pyDOE import *
from sampler import *
from qoi_functions import *
from objective_functions import *


if __name__=='__main__':
	qoi_idx = int(sys.argv[1])
	np.random.seed(1223)
	n = 10
	n_true = 10000000
	dim = 5
	noise = 0
	noise_true = 0
	sigma = eval('lambda x: ' + str(noise))
	sigma_true = eval('lambda x: ' + str(noise_true))
	objective_true = Ex4Func(sigma=sigma_true)
	objective = Ex4Func(sigma=sigma)
	X_true = lhs(dim, n_true)
	Y_true = np.array([objective(x) for x in X_true])[:, None]
	true_mean = qoi_func(Y_true, qoi=qoi_idx)
	print 'true Q[f(.)]: ', true_mean
	quit()
	a = np.array([0., 0., 0., 0., 0.])
	b = np.array([1., 1., 1., 1., 1.])
	X_init = lhs(dim, n)
	Y_init = np.array([objective(x) for x in X_init])[:, None]
	num_quad_points = 1000
	quad_points = lhs(dim, num_quad_points)
	quad_points_weight = np.ones(num_quad_points)
	num_it = 50
	out_dir = get_out_dir_name(n=n, num_it=num_it, test_id=4, qoi=qoi_idx)
	if os.path.isdir(out_dir):
		shutil.rmtree(out_dir)
	os.makedirs(out_dir)
	bounds = {"b":b, "a":a}
	hmc_priors = {"ell_kern_variance_prior_list":[gpflow.priors.Gamma(1., 1.), gpflow.priors.Gamma(1., 1.),
	gpflow.priors.Gamma(1., 1.), gpflow.priors.Gamma(1., 1.), gpflow.priors.Gamma(1., 1.)],
	"ell_kern_lengthscale_prior_list":[gpflow.priors.Gamma(1., 1.), gpflow.priors.Gamma(1., 1.),
	gpflow.priors.Gamma(1., 1.), gpflow.priors.Gamma(1., 1.), gpflow.priors.Gamma(1., 1.)], 
	"ss_kern_variance_prior_list":[gpflow.priors.Gamma(1., 1.), gpflow.priors.Gamma(1., 1.),
	gpflow.priors.Gamma(1., 1.), gpflow.priors.Gamma(1., 1.), gpflow.priors.Gamma(1., 1.)],
	"ss_kern_lengthscale_prior_list":[gpflow.priors.Gamma(1., 1.), gpflow.priors.Gamma(1., 1.),
	gpflow.priors.Gamma(1., 1.), gpflow.priors.Gamma(1., 1.), gpflow.priors.Gamma(1., 1.)], 
	"mean_func_ell_const_list":[0., 0., 0., 0., 0.],
	"mean_func_ss_const_list":[0., 0., 0., 0., 0.]}
	x_hyp = 0.6 * np.ones(dim)[None, :]
	kls = KLSampler(X_init,
                    Y_init,
                    x_hyp,
                    noisy=False,
                    qoi_func=qoi_func,
                    qoi_idx=qoi_idx,
                    obj_func=objective,
                    true_func=objective_true,
                    mcmc_from_scratch=False,
                    map_max_iter=5000, 
                    mcmc_samples=500,
                    hmc_burn=1000,
					hmc_epsilon=0.00005,
                    hmc_thin=5,
                    hmc_lmax=100,
                    nugget=1e-4,
                    jitter=1e-4,
                    kld_tol=1e-6,
                    func_name=os.path.join(out_dir, 'ex4'),
                    energy=0.95,
                    num_quad_points=num_quad_points,
                    quad_points=quad_points,
                    quad_points_weight=quad_points_weight,
                    max_it=num_it,
                    bounds=bounds,
					**hmc_priors)
	X, Y, kld, X_design, mu_qoi, sigma_qoi, models, samples = kls.optimize(num_designs=10000,
		verbose=1,
		plots=0,
		num_designs_ego=30,
		num_samp=100,
		num_samp_yhyp=20,
		num_samp_gp=20)
	np.save(os.path.join(out_dir, 'X.npy'), X)
	np.save(os.path.join(out_dir, 'Y.npy'), Y)
	np.save(os.path.join(out_dir, 'kld.npy'), kld)
	np.save(os.path.join(out_dir, 'mu_qoi.npy'), mu_qoi)
	np.save(os.path.join(out_dir, 'sigma_qoi.npy'), sigma_qoi)
	with open(os.path.join(out_dir, "models.pkl"), "wb") as f:
		pickle.dump(models, f)
	with open(os.path.join(out_dir, "samples.pkl"), "wb") as f:
		pickle.dump(samples, f)
	kld_max = np.ndarray(kld.shape[0])
	for i in xrange(kld.shape[0]):
		kld_max[i] = max(kld[i, :])
	plt.plot(np.arange(len(kld_max)), kld_max / max(kld_max), color=sns.color_palette()[1])
	plt.xticks(np.arange(0, len(kld_max), step=5), np.arange(0, len(kld_max), step=5), fontsize=12)
	plt.xlabel('iterations', fontsize=16)
	plt.ylabel('relative maximum EKLD', fontsize=16)
	plt.savefig(os.path.join(out_dir,'ekld.png'), dpi=(900), figsize=(3.25, 3.25))
	plt.clf()
	size = 10000
	x = np.ndarray((size, len(mu_qoi)))
	x_us = np.ndarray((size, len(mu_qoi)))
	x_rs = np.ndarray((size, len(mu_qoi)))
	for i in xrange(len(mu_qoi)):
		x[:, i] = norm.rvs(loc=mu_qoi[i], scale=sigma_qoi[i] ** .5, size=size)
	# 	x_us[:, i] = norm.rvs(loc=comp_log[0][i], scale=comp_log[1][i] ** .5, size=size)
	# 	x_rs[:, i] = norm.rvs(loc=comp_log[2][i], scale=comp_log[3][i] ** .5, size=size)
	bp_ekld = plt.boxplot(x, positions=np.arange(n, n + len(mu_qoi)), conf_intervals=np.array([[2.5, 97.5]] * x.shape[1]))
	pos = np.arange(n, n + len(mu_qoi))
	plt.plot(pos, true_mean * np.ones(len(pos)), '--', label='true value of the QoI', linewidth=4)
	plt.setp(bp_ekld['boxes'], color='black')
	plt.setp(bp_ekld['whiskers'], color='black')
	plt.setp(bp_ekld['caps'], color='black')
	# plt.setp(bp_ekld['medians'], color='blacksns.color_palette()[1])
	plt.setp(bp_ekld['fliers'], color=sns.color_palette()[1], marker='o')
	plt.xlabel('no. of samples', fontsize=16)
	plt.ylabel('QoI', fontsize=16)
	plt.yticks(fontsize=16)
	plt.xticks(np.arange(min(pos), max(pos) + 1, 5), np.arange(min(pos), max(pos) + 1, 5), fontsize=16)
	plt.legend(fontsize=12)
	plt.savefig(os.path.join(out_dir, 'box.png'), dpi=(900), figsize=(3.25, 3.25))
	plt.clf()
	sns.distplot(norm.rvs(loc=mu_qoi[0], scale=sigma_qoi[0] ** .5, size=size), color=sns.color_palette()[1], label='initial distribution of QoI', norm_hist=True)
	sns.distplot(norm.rvs(loc=mu_qoi[-1], scale=sigma_qoi[-1] ** .5, size=size), hist=True, color=sns.color_palette()[0], label='final distribution of QoI', norm_hist=True)
	# plt.scatter(np.mean(Y_u), 0, c=sns.color_palette()[2], label='uncertainty sampling mean')
	plt.scatter(true_mean, 0, c=sns.color_palette()[2], label='true mean')
	# plt.xticks([mu_qoi[-1]])
	plt.legend(fontsize=12)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.xlabel('QoI', fontsize=16)
	plt.ylabel('p(QoI)', fontsize=16)
	plt.savefig(os.path.join(out_dir, 'dist.png'), dpi=(900), figsize=(3.25, 3.25))
	plt.clf()
	# Comparison plot
	bp_ekld = plt.boxplot(x, positions=np.arange(n, n + len(mu_qoi)), conf_intervals=np.array([[2.5, 97.5]] * x.shape[1]))
	plt.setp(bp_ekld['boxes'], color=sns.color_palette()[1])
	plt.setp(bp_ekld['whiskers'], color=sns.color_palette()[1])
	plt.setp(bp_ekld['caps'], color=sns.color_palette()[1])
	plt.setp(bp_ekld['medians'], color=sns.color_palette()[1])
	# ekld_fl = plt.setp(bp_ekld['fliers'], color=sns.color_palette()[1], marker='o')
	# bp_us = plt.boxplot(x_us, positions=np.arange(n, n + len(mu_qoi)), conf_intervals=np.array([[2.5, 97.5]] * x.shape[1]))
	# plt.setp(bp_us['boxes'], color=sns.color_palette()[2])
	# plt.setp(bp_us['whiskers'], color=sns.color_palette()[2])
	# plt.setp(bp_us['caps'], color=sns.color_palette()[2])
	# plt.setp(bp_us['medians'], color=sns.color_palette()[2])
	# # us_fl = plt.setp(bp_us['fliers'], color=sns.color_palette()[2], marker='x')
	# bp_rs = plt.boxplot(x_rs, positions=np.arange(n, n + len(mu_qoi)), conf_intervals=np.array([[2.5, 97.5]] * x.shape[1]))
	# plt.setp(bp_rs['boxes'], color=sns.color_palette()[3])
	# plt.setp(bp_rs['whiskers'], color=sns.color_palette()[3])
	# plt.setp(bp_rs['caps'], color=sns.color_palette()[3])
	# plt.setp(bp_rs['medians'], color=sns.color_palette()[3])
	# rs_fl = plt.setp(bp_rs['fliers'], color=sns.color_palette()[3], marker='*')
	hekld, = plt.plot([1, 1], color=sns.color_palette()[1])
	# hus, = plt.plot([1, 1], color=sns.color_palette()[2])
	# hur, = plt.plot([1, 1], color=sns.color_palette()[3])

	# plt.scatter(pos, comp_log[0], s=40, marker='x', color=sns.color_palette()[2], label='uncertainty sampling')
	# plt.scatter(pos, comp_log[2], s=30, marker='*', color=sns.color_palette()[3], label='random sampling')
	# plt.scatter(pos, mu_qoi, s=20, marker='o', color=sns.color_palette()[1], label='EKLD')
	plt.plot(pos, true_mean * np.ones(len(pos)), '--', label='true value of the QoI')
	plt.xlabel('no. of samples')
	plt.ylabel('QoI')
	plt.xticks(np.arange(min(pos), max(pos) + 1, 5), np.arange(min(pos), max(pos) + 1, 5), fontsize=4)
	# plt.legend((hekld, hus, hur), ('EKLD', 'uncertainty sampling', 'random sampling'))
	hekld.set_visible(False)
	# hus.set_visible(False)
	# hur.set_visible(False)
	# plt.ylim(np.min(np.vstack([x, x_us, x_rs])), np.max(np.hstack([x, x_us, x_rs])))
	plt.savefig(os.path.join(out_dir, 'comparison.png'), dpi=(900), figsize=(3.25, 3.25))
	quit()
