import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord,Galactocentric,ICRS
from .priors import determineprior,defaultprior

import jax
import jax.numpy as jnp

import numpyro
from numpyro import distributions as dist, infer
from numpyro.distributions import MixtureGeneral

def cluster_model_3d(covar, par={}, priors={}, additionalinfo={}):
    
    starpos = jnp.vstack([par['RA'],par['Dec'],par['PM_RA'],par['PM_Dec'],par['dist'],par['RV']])
    
    # figure out how many stars
    nstars = len(par['vrad'])

	# sample from priors
    sample = {}
    for pp in ['X','Y','Z','v_X','v_Y','v_Z']:
        if pp in priors.keys():
            sample[pp] = determineprior(pp,priors[pp])
            sample['sigma_'+pp] = determineprior('sigma_'+pp,priors['sigma_'+pp])
        else:
            sample[pp] = defaultprior(pp)
            sample['sigma_'+pp] = defaultprior('sigma_'+pp)
        
        if pp+'_bg' in priors.keys():
            sample[pp+'_bg'] = determineprior(pp+'_bg',priors[pp+'_bg'])
            sample['sigma_'+pp+'_bg'] = determineprior('sigma_'+pp+'_bg',priors['sigma_'+pp+'_bg'])
        else:
            sample[pp+'_bg'] = defaultprior(pp+'_bg')
            sample['sigma_'+pp+'_bg'] = defaultprior('sigma_'+pp+'_bg')
    
    if 'Q' in priors.keys():
        sample['Q'] = determineprior('Q',priors['Q'])
    else:
        sample['Q'] = defaultprior('Q')
    
    
    with numpyro.plate("stars", nstars):

        # The background distance distribution 
        pos_fg = [sample['X'],sample['Y'],sample['Z'],sample['v_X'],sample['v_Y'],sample['v_Z']]
        covarFG = jnp.zeros([len(pos_fg),len(pos_fg)],dtype=float)
        for ii,pp in enumerate(['X','Y','Z','v_X','v_Y','v_Z']):
            covarFG[ii,ii] = np.sqrt(sample['sigma_'+pp])
        
        dist_fg = dist.MultivariateNormal(loc=pos_fg, covariance_matrix=covarFG)

        # The foreground distribution 
        pos_bg = [sample['X_bg'],sample['Y_bg'],sample['Z_bg'],sample['v_X_bg'],sample['v_Y_bg'],sample['v_Z_bg']]
        covarBG = jnp.zeros([len(pos_bg),len(pos_bg)],dtype=float)
        for ii,pp in enumerate(['X','Y','Z','v_X','v_Y','v_Z']):
            covarBG[ii,ii] = np.sqrt(sample['sigma_'+pp+'_bg'])
        
        dist_bg = dist.MultivariateNormal(loc=pos_bg, covariance_matrix=covarBG)
        
        # Now we "mix" the foreground and background distributions using the
        # "cluster membership fraction" parameter to specify the mixing weights.
        mixture = MixtureGeneral(
            dist.Categorical(probs=jnp.stack((sample['Q'], 1 - sample['Q']), axis=-1)),
            [dist_fg, dist_bg],
        )
        r = numpyro.sample("r", mixture)
        
        # convert r to observed coordinates
        sc = SkyCoord(
            x=r[0]*u.kpc,
            y=r[1]*u.kpc,
            z=r[2]*u.kpc,
            v_x=r[3]*u.km/u.s,
            v_y=r[4]*u.km/u.s,
            v_z=r[5]*u.km/u.s,
            frame=Galactocentric)
        sc.transform_to(ICRS)
        ra        = float(sc.ra.value)
        dec       = float(sc.dec.value)
        pmra      = float(sc.pm_ra_cosdec.value)
        pmdec     = float(sc.pm_dec.value)
        distance  = float(sc.distance.value)
        rv        = float(sc.radial_velocity.value)
        
        r_t = [ra,dec,pmra,pmdec,distance,rv]
        
        log_probs = mixture.component_log_probs(r)
        numpyro.deterministic(
            "p", log_probs - jax.nn.logsumexp(log_probs, axis=-1, keepdims=True)
        )

        # Finally, we convert the distance to parallax and add the zero-point offset.
        numpyro.sample("starpos", dist.MultivariateNormal(loc=r_t, covariance_matrix=covar), obs=starpos)
