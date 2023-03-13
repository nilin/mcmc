import jax
import jax.numpy as jnp
import jax.random as rnd
import typing
from dataclasses import dataclass
from functools import partial
from session import nextkeys


# there has to be a A[:,None...None]*B way of doing this:
mult=jax.vmap(jnp.multiply,in_axes=(0,0))




@dataclass
class Metropolis:

    densities=None
    key=rnd.PRNGKey(0)

    def __init__(self,P,proposal,walkers):
        self.walkers=walkers
        self.P=P
        self.__sample__=jax.jit(partial(self._sample_,P,proposal))

    @staticmethod
    def _sample_(P,proposal,key,params,walkers,densities):
        key1,key2=rnd.split(key)
        proposals=proposal(key1,walkers)
        newdensities=P(params,proposals)
        a=newdensities/densities
        u=rnd.uniform(key2,a.shape,minval=0,maxval=1)
        accepted=(u<a)

        walkers=mult(proposals,accepted)+mult(walkers,(1-accepted))
        densities=mult(newdensities,accepted)+mult(densities,(1-accepted))
        return walkers,densities
    
    def sample(self,params,steps=1):
        if self.densities is None:
            self.densities=self.P(params,self.walkers)
        for key in nextkeys(steps):
            self.walkers,self.densities=self.__sample__(key,params,self.walkers,self.densities)
        return self.walkers

        



gaussianproposal_=lambda key,X,std: X+rnd.normal(key,X.shape)*std
gaussianproposal=lambda std: partial(gaussianproposal_,std=std)
