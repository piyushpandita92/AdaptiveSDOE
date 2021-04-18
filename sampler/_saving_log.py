"""
Write to a file on the fly.
"""
import h5py
from pandas import HDFStore

class WriteToFile(object):
	"""
	write the current state to file.
	"""
	def __init__(self, filename='data_file.h5'):
		self.filename = filename
		self.hdf = HDFStore(self.filename)

	def write_to_file(self, data, data_name):
		"""
		write to file
		"""
		self.hdf[data_name] = data