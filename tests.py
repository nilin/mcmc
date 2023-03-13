import jax
import jax.numpy as jnp
import jax.random as rnd
import sys
import matplotlib.pyplot as plt
import seaborn
import jax, jax.numpy as jnp
import flax
from session import nextkey
import mcmc

if 'db' in sys.argv:
    print('jit disabled')
    from jax.config import config as jconf
    jconf.update('jax_disable_jit', True)


class Model(flax.linen.Module):

    @flax.linen.compact
    def __call__(self,X):
        Y=flax.linen.Dense(features=100)(X)
        Y=jax.nn.gelu(Y)
        Y=flax.linen.Dense(features=100)(X)
        Y=jax.nn.gelu(Y)
        Y=flax.linen.Dense(features=100)(X)
        Y=jax.nn.gelu(Y)
        Y=flax.linen.Dense(features=1)(Y)
        Y=jnp.squeeze(Y)
        #return jnp.ones_like(Y),Y#-(jnp.abs(jnp.squeeze(X))<1)*inf
        return jnp.sign(Y), jnp.log(jnp.abs(Y))
    

if __name__=='__main__':
    P_test_1=lambda x: jnp.squeeze(jnp.abs(x)*(jnp.abs(x)<1))
    P_test_2=lambda x: P_test_1(x[:,0])*jnp.exp(-(x[:,1]-1)**2/2)
    X_init_test=rnd.uniform(nextkey(),(1000,2),minval=-1,maxval=1)

    X=X_init_test
    sampler=mcmc.Metropolis(P=lambda _,X:P_test_2(X),proposal=mcmc.gaussianproposal(.5),walkers=X)
    sampler.sample(None,steps=1000)
    X=jnp.concatenate([sampler.sample(None) for i in range(1000)],axis=0)
    seaborn.kdeplot(X,bw_method=.05)
    plt.show()