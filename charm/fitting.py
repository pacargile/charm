import sys
import dynesty
from datetime import datetime
import numpy as np

from .model import clustermodel
from .priors import priors

class charmfit(object):
	"""docstring for charmfit"""
	def __init__(self,inarr,*args,**kwargs):
		super(charmfit, self).__init__()
		
		# input array
		self.inarr = inarr

		# other input args and keywords
		self.args = args
		self.kwargs = kwargs

		# set ndim
		self.ndim = 6

		# set verbose
		self.verbose = self.kwargs.get('verbose',True)

		# make sure inarr has correct parameters
		try:
			assert('RA' in self.inarr.keys())
			assert('Dec' in self.inarr.keys())
			assert('Parallax' in self.inarr.keys())
		except AssertionError:
			print('-- Could not find RA/Dec/Parallax keys in input dictionary')
			raise IOError

		# initialize the model class
		self.clustermodel = clustermodel(
			self.inarr,
			self.kwargs.get('Nsamples',1000.0),
			modeltype=self.kwargs.get('ModelType','gaussian'))


		# initialize the prior class
		priordict = self.kwargs.get('priordict',{})
		self.priors = priors
		self.priorobj = self.priors(priordict)

		# initialize the output file
		self._initoutput()

		# bulid sampler
		self._buildsampler()


	def _initoutput(self):
		# determine if user defined output filename
		output_fn = self.kwargs.get('output','test.out')

		# init output file
		self.outff = open(output_fn,'w')
		self.outff.write('Iter ')

		self.outff.write('X sig_X Y sig_Y Z sig_Z ')		

		self.outff.write('log(lk) log(vol) log(wt) h nc log(z) delta(log(z))')
		self.outff.write('\n')

	def _buildsampler(self):
		# pull out user defined sampler variables
		samplerdict = self.kwargs.get('samplerdict',{})
		self.npoints = samplerdict.get('npoints',200)
		self.samplertype = samplerdict.get('samplertype','multi')
		self.bootstrap = samplerdict.get('bootstrap',0)
		self.update_interval = samplerdict.get('update_interval',0.6)
		self.samplemethod = samplerdict.get('samplemethod','unif')
		self.delta_logz_final = samplerdict.get('delta_logz_final',1.0)
		self.flushnum = samplerdict.get('flushnum',100)
		self.maxiter = samplerdict.get('maxiter',sys.maxsize)

		# initialize sampler object
		self.dy_sampler = dynesty.NestedSampler(
			self.likefn,
			self.priorobj.priortrans,
			self.ndim,
			# logl_args=[self.likeobj,self.priorobj],
			nlive=self.npoints,
			bound=self.samplertype,
			sample=self.samplemethod,
			update_interval=self.update_interval,
			bootstrap=self.bootstrap,
			)

	def likefn(self,args):
		likeprob = self.clustermodel.likefn(args)
		# likeprob += -0.5*(((args[1]-20.0)/5.0)**2.0)
		return likeprob

	def runsampler(self):

		# set start time
		starttime = datetime.now()
		if self.verbose:
			print(
				('Start Dynesty w/ {0} number of samples, Ndim = {1}, '
				'and w/ stopping criteria of dlog(z) = {2}: {3}').format(
					self.npoints,self.ndim,self.delta_logz_final,starttime))
		sys.stdout.flush()
		
		ncall = 0
		nit = 0

		iter_starttime = datetime.now()
		deltaitertime_arr = []

		# start sampling
		for it, results in enumerate(self.dy_sampler.sample(dlogz=self.delta_logz_final)):
			(worst, ustar, vstar, loglstar, logvol, logwt, logz, logzvar,
				h, nc, worst_it, propidx, propiter, eff, delta_logz) = results			

			self.outff.write('{0} '.format(it))
			self.outff.write(' '.join([str(q) for q in vstar]))
			self.outff.write(' {0} {1} {2} {3} {4} {5} {6} '.format(
				loglstar,logvol,logwt,h,nc,logz,delta_logz))
			self.outff.write('\n')

			ncall += nc
			nit = it

			deltaitertime_arr.append((datetime.now()-iter_starttime).total_seconds()/float(nc))
			iter_starttime = datetime.now()

			if ((it%self.flushnum) == 0) or (it == self.maxiter):
				self.outff.flush()

				if self.verbose:
					# format/output results
					if logz < -1e6:
						logz = -np.inf
					if delta_logz > 1e6:
						delta_logz = np.inf
					if logzvar >= 0.:
						logzerr = np.sqrt(logzvar)
					else:
						logzerr = np.nan
					if logzerr > 1e6:
						logzerr = np.inf
						
					sys.stdout.write("\riter: {0:d} | nc: {1:d} | ncall: {2:d} | eff(%): {3:6.3f} | "
						"logz: {4:6.3f} +/- {5:6.3f} | dlogz: {6:6.3f} > {7:6.3f}   | mean(time):  {8}  "
						.format(nit, nc, ncall, eff, 
							logz, logzerr, delta_logz, 
							self.delta_logz_final,np.mean(deltaitertime_arr)))
					sys.stdout.flush()
					deltaitertime_arr = []
			if (it == self.maxiter):
				break

		# add live points to sampler object
		for it2, results in enumerate(self.dy_sampler.add_live_points()):
			# split up results
			(worst, ustar, vstar, loglstar, logvol, logwt, logz, logzvar,
			h, nc, worst_it, boundidx, bounditer, eff, delta_logz) = results

			self.outff.write('{0} '.format(nit+it2))

			self.outff.write(' '.join([str(q) for q in vstar]))
			self.outff.write(' {0} {1} {2} {3} {4} {5} {6} '.format(
				loglstar,logvol,logwt,h,nc,logz,delta_logz))
			self.outff.write('\n')

			ncall += nc

			if self.verbose:
				# format/output results
				if logz < -1e6:
					logz = -np.inf
				if delta_logz > 1e6:
					delta_logz = np.inf
				if logzvar >= 0.:
					logzerr = np.sqrt(logzvar)
				else:
					logzerr = np.nan
				if logzerr > 1e6:
					logzerr = np.inf
				sys.stdout.write("\riter: {:d} | nc: {:d} | ncall: {:d} | eff(%): {:6.3f} | "
					"logz: {:6.3f} +/- {:6.3f} | dlogz: {:6.3f} > {:6.3f}      "
					.format(nit + it2, nc, ncall, eff, 
						logz, logzerr, delta_logz, self.delta_logz_final))

				sys.stdout.flush()
		
		# close the output file
		self.outff.close()
		sys.stdout.write('\n')

		finishtime = datetime.now()
		if self.verbose:
			print('RUN TIME: {0}'.format(finishtime-starttime))
			sys.stdout.flush()

