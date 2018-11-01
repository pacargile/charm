import numpy as np
from scipy.stats import norm as gaussian
from astropy import units as u
from astropy.coordinates import SkyCoord

def gauss(args):
	x,mu,sigma = args
	return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x - mu)**2 / (2*sigma**2))

class clustermodel(object):
	"""docstring for clustermodel"""
	def __init__(self, inarr, Nsamp, modeltype='gaussian'):
		super(clustermodel, self).__init__()
		self.inarr = inarr
		self.Nstars = len(self.inarr)
		self.starid = range(self.Nstars)
		self.Nsamp = Nsamp
		self.modeltype = modeltype

		# generate grid of samples for each star
		# self.starsamples = np.empty( (self.Nstars, self.Nsamp) )
		# for idx in self.starid:
		# 	self.starsamples[idx,:] = gaussian(
		# 		loc=self.inarr['Parallax'][idx], 
		# 		scale=self.inarr['Parallax_Error'][idx]).rvs(size=self.Nsamp)

		"""
		self.starsamples = np.array([{} for _ in range(self.Nstars)])
		for idx in self.starid:
			RAdist = gaussian(
				loc=self.inarr['RA'][idx], 
				scale=self.inarr['RA_Error'][idx]).rvs(size=self.Nsamp)
			Decdist = gaussian(
				loc=self.inarr['Dec'][idx], 
				scale=self.inarr['Dec_Error'][idx]).rvs(size=self.Nsamp)
			Distdist = 1000.0/gaussian(
				loc=self.inarr['Parallax'][idx], 
				scale=self.inarr['Parallax_Error'][idx]).rvs(size=self.Nsamp)

			Xarr = []
			Yarr = []
			Zarr = []
			for ra_i,dec_i,dist_i in zip(RAdist,Decdist,Distdist):
				c = SkyCoord(ra=ra_i*u.deg,dec=dec_i*u.deg,distance=dist_i*u.pc)
				Xarr.append(float(c.galactocentric.x.value))
				Yarr.append(float(c.galactocentric.y.value))
				Zarr.append(float(c.galactocentric.z.value))

			self.starsamples[idx] = ({
				'X':np.array(Xarr),
				'Y':np.array(Yarr),
				'Z':np.array(Zarr),
				})
		"""
		self.starsamples = np.empty((3,self.Nstars,self.Nsamp))
		for idx in range(self.Nstars):
			RAdist = gaussian(
				loc=self.inarr['RA'][idx], 
				scale=self.inarr['RA_Error'][idx]).rvs(size=self.Nsamp)
			Decdist = gaussian(
				loc=self.inarr['Dec'][idx], 
				scale=self.inarr['Dec_Error'][idx]).rvs(size=self.Nsamp)
			Distdist = 1000.0/gaussian(
				loc=self.inarr['Parallax'][idx], 
				scale=self.inarr['Parallax_Error'][idx]).rvs(size=self.Nsamp)

			for idd,dim in enumerate(['x','y','z']):
				c = SkyCoord(ra=RAdist*u.deg,dec=Decdist*u.deg,distance=Distdist*u.pc)
				if dim == 'x':
					self.starsamples[idd,idx,:] = np.array(c.galactocentric.x.value)
				elif dim == 'y':
					self.starsamples[idd,idx,:] = np.array(c.galactocentric.y.value)
				elif dim == 'z':
					self.starsamples[idd,idx,:] = np.array(c.galactocentric.z.value)
				else:
					raise IOError

	def likefn(self,arg):
		# dist,sigma_dist = arg
		x,sigma_x,y,sigma_y,z,sigma_z = arg

		# calculate like for all stars

		if self.modeltype == 'gaussian':
			# Gaussian model
			like = (
				((1.0/(np.sqrt(2.0*np.pi)*sigma_x)) * 
					np.exp( -0.5 * ((self.starsamples[0,...] -x)**2.0)*(sigma_x**-2.0) )) + 

				((1.0/(np.sqrt(2.0*np.pi)*sigma_y)) * 
					np.exp( -0.5 * ((self.starsamples[1,...]-y)**2.0)*(sigma_y**-2.0) )) + 

				((1.0/(np.sqrt(2.0*np.pi)*sigma_z)) * 
					np.exp( -0.5 * ((self.starsamples[2,...]-z)**2.0)*(sigma_z**-2.0) )) 
				)

		elif self.modeltype == 'cauchy':
			# Cauchy model
			like = (
				((1.0/(np.pi*sigma_x)) *
					(sigma_x**2.0)/( ((self.starsamples[0,...]-x)**2.0) + (sigma_x**2.0) )) + 

				((1.0/(np.pi*sigma_y)) *
					(sigma_y**2.0)/( ((self.starsamples[1,...]-y)**2.0) + (sigma_y**2.0) )) + 

				((1.0/(np.pi*sigma_z)) *
					(sigma_z**2.0)/( ((self.starsamples[2,...]-z)**2.0) + (sigma_z**2.0) )) 
				)

		elif self.modeltype == 'plummer':
			# Plummer model

			like = (
				( (1.0/(sigma_x**3.0)) *
					((1.0 + (((self.starsamples[0,...]-x)/sigma_x)**2.0))**(-5.0/2.0)) ) + 

				( (1.0/(sigma_y**3.0)) *
					((1.0 + (((self.starsamples[1,...]-y)/sigma_y)**2.0))**(-5.0/2.0)) ) + 

				( (1.0/(sigma_z**3.0)) *
					((1.0 + (((self.starsamples[2,...]-z)/sigma_z)**2.0))**(-5.0/2.0)) ) 
				)

		else:
			print('Did not understand model type')
			raise IOError

		if np.min(like) <= np.finfo(np.float).eps:
			return -np.inf

		like = (like.T*self.inarr['Prob']).T
		lnp = np.sum(np.log(like))
		return lnp