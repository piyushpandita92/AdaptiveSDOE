import os
import sys
import numpy as np
__all__ = ['qoi_func', 'get_out_dir_name', 'get_out_comp_dir_name']
	
def get_out_dir_name(n, num_it, test_id, qoi=1, comp=False):
	if qoi == 1:
		if comp:
			out_dir = 'comp_mcmc_mean_ex{0:d}_n={1:d}_it={2:d}'.format(test_id, n, num_it)
		else:
			out_dir = 'mcmc_mean_ex{0:d}_n={1:d}_it={2:d}'.format(test_id, n, num_it)
	elif qoi == 2:
		if comp:
			out_dir = 'comp_mcmc_var_ex{0:d}_n={1:d}_it={2:d}'.format(test_id, n, num_it)
		else:
			out_dir = 'mcmc_var_ex{0:d}_n={1:d}_it={2:d}'.format(test_id, n, num_it)
	elif qoi == 3:
		if comp:
			out_dir = 'comp_mcmc_max_ex{0:d}_n={1:d}_it={2:d}'.format(test_id, n, num_it)
		else:
			out_dir = 'mcmc_max_ex{0:d}_n={1:d}_it={2:d}'.format(test_id, n, num_it)
	elif qoi == 4:
		if comp:
			out_dir = 'comp_mcmc_min_ex{0:d}_n={1:d}_it={2:d}'.format(test_id, n, num_it)
		else:
			out_dir = 'mcmc_min_ex{0:d}_n={1:d}_it={2:d}'.format(test_id, n, num_it)
	elif qoi == 5:
		if comp:
			out_dir = 'comp_mcmc_pf_ex{0:d}_n={1:d}_it={2:d}'.format(test_id, n, num_it)
		else:
			out_dir = 'mcmc_pf_ex{0:d}_n={1:d}_it={2:d}'.format(test_id, n, num_it)
	return out_dir

def get_out_comp_dir_name(n, num_it, test_id, qoi=1, iaf_name='ei', comp=False):
	if qoi == 1:
		qoi = 'mean'
	elif qoi == 2:
		qoi = 'var'
	elif qoi == 3:
		qoi = 'max'
	elif qoi == 4:
		qoi = 'min'
	elif qoi == 5:
		qoi = 'pf'
	out_dir = 'comp_mcmc_' + qoi + '_' + iaf_name + '_ex{0:d}_n={1:d}_it={2:d}'.format(test_id, n, num_it)
	return out_dir

def qoi_func(f, qoi=1, percentile=97.5):
	"""
	An example QoI, the maximum of a function.
	:param f:   array like consisting of values of a sample of the underlying function
	:return:    a scalar value
	"""
	if qoi == 1:
		return np.mean(f, axis=0)
	elif qoi == 2:
		return np.var(f, axis=0)
	elif qoi == 3:
		return np.amax(f, axis=0)
	elif qoi == 4:
		return np.amin(f, axis=0)
	elif qoi == 5:
		# Probability of f being less than ```percentile``` is what we are after.
		return np.percentile(f, percentile, axis=0)
		# f[f>threshold] = 0
		# f[f!=0] = 1
		# return np.mean(f, axis=0)