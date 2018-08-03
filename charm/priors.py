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
		udist,usig_dist = upars

		outarr = []

		if 'Dist' in self.priordict.keys():
			dist = (
				(max(self.priordict['Dist'])-min(self.priordict['Dist']))*udist + 
				min(self.priordict['Dist'])
				)
		else:
			dist = (1e+5 - 1e+2)*udist + 1e+2

		outarr.append(dist)

		if 'sig_Dist' in self.priordict.keys():
			sig_dist = (
				(max(self.priordict['sig_Dist'])-min(self.priordict['sig_Dist']))*usig_dist + 
				min(self.priordict['sig_Dist'])
				)
		else:
			sig_dist = (100.0 - 0.0)*usig_dist + 0.0

		outarr.append(sig_dist)

		return outarr
