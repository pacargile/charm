import numpyro
from numpyro.infer import MCMC, NUTS,initialization

from jax import jit,lax,jacfwd
from jax import random as jrandom
import jax.numpy as jnp

from datetime import datetime
import sys,os
from astropy.table import Table

from .model import cluster_model_3d

class charmfit(object):
    """docstring for charmfit"""
    def __init__(self,inarr,*args,**kwargs):
        super(charmfit, self).__init__()
        
        self.rng_key = jrandom.PRNGKey(0)

    def run(self,indict):
        
        starttime = datetime.now()
        
        sampler = MCMC(
			NUTS(cluster_model_3d),
			num_warmup=200,
			num_samples=1000,
			num_chains=1,
			progress_bar=True,
		)
        
        sampler.run(self.rng_key, indict['covar'], par=indict['pars'], priors=indict['priors'])

        posterior = sampler.get_samples()
