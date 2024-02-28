import numpyro
import numpyro.distributions as distfn
import jax.numpy as jnp

def defaultprior(parname):
    # define defaults for sampled parameters
    if parname == "X":
        return numpyro.sample("X", distfn.Uniform(0,200))
    if parname == "Y":
        return numpyro.sample("Y", distfn.Uniform(0,200))
    if parname == "Z":
        return numpyro.sample("Z", distfn.Uniform(0,200))

    if parname == "X_bg":
        return numpyro.sample("X_bg", distfn.Uniform(0,200))
    if parname == "Y_bg":
        return numpyro.sample("Y_bg", distfn.Uniform(0,200))
    if parname == "Z_bg":
        return numpyro.sample("Z_bg", distfn.Uniform(0,200))

    if parname == "sigma_X":
        return numpyro.sample("sigma_X", distfn.Uniform(0,100))
    if parname == "sigma_Y":
        return numpyro.sample("sigma_Y", distfn.Uniform(0,100))
    if parname == "sigma_Z":
        return numpyro.sample("sigma_Z", distfn.Uniform(0,100))

    if parname == "sigma_X_bg":
        return numpyro.sample("sigma_X_bg", distfn.Uniform(0,100))
    if parname == "sigma_Y_bg":
        return numpyro.sample("sigma_Y_bg", distfn.Uniform(0,100))
    if parname == "sigma_Z_bg":
        return numpyro.sample("sigma_Z_bg", distfn.Uniform(0,100))

def determineprior(parname,priorinfo,*args):

    # standard prior distributions
    if priorinfo[0] == 'uniform':
        return numpyro.sample(parname,distfn.Uniform(*priorinfo[1]))
    if priorinfo[0] == 'normal':
        return numpyro.sample(parname,distfn.Normal(*priorinfo[1]))
    if priorinfo[0] == 'halfnormal':
        return numpyro.sample(parname,distfn.HalfNormal(priorinfo[1]))
    if priorinfo[0] == 'tnormal':
        return numpyro.sample(parname,distfn.TruncatedDistribution(
            distfn.Normal(loc=priorinfo[1][0],scale=priorinfo[1][1]),
            low=priorinfo[1][2],high=priorinfo[1][3]))
        
    if priorinfo[0] == 'fixed':
        return numpyro.deterministic(parname,priorinfo[1])
