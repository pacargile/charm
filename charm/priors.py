import numpy as np

class priors(object):
	"""docstring for priors"""
	def __init__(self, inpriordict):
		super(priors, self).__init__()

		# find uniform priors and put them into a 
		# dictionary used for the prior transformation
		self.priordict = {}

		# put any additional priors into a dictionary so that
		# they can be applied in the lnprior_* functions
		self.additionalpriors = {}

		for kk in inpriordict.keys():
			for ii in inpriordict[kk].keys():
				if ii == 'uniform':
					self.priordict[kk] = inpriordict[kk]['uniform']
				else:
					try:
						self.additionalpriors[kk][ii] = inpriordict[kk][ii]
					except KeyError:
						self.additionalpriors[kk] = {ii:inpriordict[kk][ii]}

	def priortrans(self,upars):
		# split upars
		ux,usig_x,uy,usig_y,uz,usig_z = upars

		outarr = []

		if 'X' in self.priordict.keys():
			x = (
				(max(self.priordict['X'])-min(self.priordict['X']))*ux + 
				min(self.priordict['X'])
				)
		else:
			x = (1e+5 - -1e+5)*ux - 1e+5

		if 'sig_X' in self.priordict.keys():
			sig_x = (
				(max(self.priordict['sig_X'])-min(self.priordict['sig_X']))*usig_x + 
				min(self.priordict['sig_X'])
				)
		else:
			sig_x = (10000.0 - 0.0)*usig_x + 0.0

		if 'Y' in self.priordict.keys():
			y = (
				(max(self.priordict['Y'])-min(self.priordict['Y']))*uy + 
				min(self.priordict['Y'])
				)
		else:
			y = (1e+5 - -1e+5)*uy - 1e+5

		if 'sig_Y' in self.priordict.keys():
			sig_y = (
				(max(self.priordict['sig_Y'])-min(self.priordict['sig_Y']))*usig_y + 
				min(self.priordict['sig_Y'])
				)
		else:
			sig_y = (10000.0 - 0.0)*usig_y + 0.0

		if 'Z' in self.priordict.keys():
			z = (
				(max(self.priordict['Z'])-min(self.priordict['Z']))*uz + 
				min(self.priordict['Z'])
				)
		else:
			z = (1e+5 - -1e+5)*uz - 1e+5

		if 'sig_Z' in self.priordict.keys():
			sig_z = (
				(max(self.priordict['sig_Z'])-min(self.priordict['sig_Z']))*usig_z + 
				min(self.priordict['sig_Z'])
				)
		else:
			sig_z = (10000.0 - 0.0)*usig_z + 0.0

		pars = [x,sig_x,y,sig_y,z,sig_z]

		return pars
