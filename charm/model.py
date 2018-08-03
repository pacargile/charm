import numpy as np
from scipy.stats import norm as gaussian

def gauss(args):
	x,mu,sigma = args
	return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x - mu)**2 / (2*sigma**2))

class clustermodel(object):
	"""docstring for clustermodel"""
	def __init__(self, inarr, Nsamp):
		super(clustermodel, self).__init__()
		self.inarr = inarr
		self.Nstars = len(self.inarr)
		self.starid = range(self.Nstars)
		self.Nsamp = Nsamp

		# generate grid of samples for each star
		self.starsamples = np.empty( (self.Nstars, self.Nsamp) )
		for idx in self.starid:
			self.starsamples[idx,:] = gaussian(
				loc=self.inarr['Parallax'][idx], 
				scale=self.inarr['Parallax_Error'][idx]).rvs(size=self.Nsamp)

	def likefn(self,arg):
		dist,sigma_dist = arg

		# calculate like for all stars

		# Gaussian model
		like = (
			(1.0/(np.sqrt(2.0*np.pi)*sigma_dist)) * 
			np.exp( -0.5 * (((1000.0/self.starsamples)-dist)**2.0)*(sigma_dist**-2.0) )
			)
		"""

		# Cauchy model
		like = (
			(1.0/(np.pi*sigma_dist)) *
			(sigma_dist**2.0)/( (((1000.0/self.starsamples)-dist)**2.0) + (sigma_dist**2.0) )
			)
		"""
		
		if np.min(like) <= np.finfo(np.float).eps:
			return -np.inf

		like = (like.T*self.inarr['Prob']).T
		lnp = np.sum(np.log(like))
		return lnp