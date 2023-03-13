import jax
import jax.numpy as jnp
import jax.random as rnd
import numpy as np
import sys
from functools import partial
import matplotlib.pyplot as plt
import seaborn
import time
import jax, jax.numpy as jnp
import flax
from functools import partial
import optax
from flax.training.train_state import TrainState
from session import nextkey
import mcmc, session
import energy
from tqdm import tqdm

if 'db' in sys.argv:
    print('jit disabled')
    from jax.config import config as jconf
    jconf.update('jax_disable_jit', True)

P_test_1=lambda x: jnp.squeeze(jnp.abs(x)*(jnp.abs(x)<1))
P_test_2=lambda x: P_test_1(x[:,0])*jnp.exp(-(x[:,1]-1)**2/2)
X_init_test=rnd.uniform(nextkey(),(1000,2),minval=-1,maxval=1)

if 't' in sys.argv:
    X=X_init_test
    sampler=mcmc.Metropolis(P=lambda _,X:P_test_2(X),proposal=mcmc.gaussianproposal(.5),walkers=X)
    sampler.sample(None,steps=1000)
    X=jnp.concatenate([sampler.sample(None) for i in range(1000)],axis=0)
    seaborn.kdeplot(X,bw_method=.05)
    plt.show()



####################################################################################################



mode='nomcmc' if 'nm' in sys.argv else 'mcmc'

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
        inf=1000
        #return jnp.ones_like(Y),Y#-(jnp.abs(jnp.squeeze(X))<1)*inf

        return jnp.sign(Y), jnp.log(jnp.abs(Y))
    


model=Model()
slog_Si=model.apply
Si_=energy.noslog(slog_Si)
Si=lambda P,X: Si_(P,X)*jnp.squeeze(jnp.abs(X)<1)
P_Si=lambda P,X: Si(P,X)**2

X=rnd.uniform(nextkey(),(1000,1),minval=-1,maxval=1)
Y=rnd.uniform(nextkey(),(1000,1),minval=-1,maxval=1)
params0=model.init(nextkey(),X)

fi=lambda x: jnp.squeeze((jnp.sin(5*x)+1)*(jnp.abs(x)<1))
#fi=lambda x: jnp.squeeze(x*(jnp.abs(x)<1))
#fi=partial(Si,params0)
P_Fi=lambda _,x: fi(x)**2

state=TrainState.create(apply_fn=slog_Si,params=params0,tx=optax.sgd(.1,.2))
vals=[]
nburn=1000
proposal=mcmc.gaussianproposal(.5)

match mode:
    case 'mcmc':
        E_value_and_grad=energy.E_value_and_grad(slog_Si,fi=fi)

        si_sampler=mcmc.Metropolis(P=P_Si,proposal=proposal,walkers=X)
        fi_sampler=mcmc.Metropolis(P=P_Fi,proposal=proposal,walkers=Y)

        X=si_sampler.sample(params0,steps=nburn)
        Y=fi_sampler.sample(None,steps=nburn)

        for i in range(1000):
            X=si_sampler.sample(state.params,steps=100)
            Y=fi_sampler.sample(None)
            
            loss,grads=E_value_and_grad(state.params,X,Y)
            state=state.apply_gradients(grads=grads)

            vals.append(-loss)
            print(-loss)

        session.save(vals,'data/mcmc')


    case 'nomcmc':
        E_value_and_grad=energy.nomcmc_value_and_grad(slog_Si,fi)

        fi_sampler=mcmc.Metropolis(P=P_Fi,proposal=proposal,walkers=Y)
        Y=fi_sampler.sample(None,steps=nburn)

        for i in range(1000):
            Y=fi_sampler.sample(None)
            
            loss,grads=E_value_and_grad(state.params,X,Y)
            state=state.apply_gradients(grads=grads)

            vals.append(-loss)
            print(-loss)

        session.save(vals,'data/nomcmc')

#x=jnp.expand_dims(jnp.arange(-2,2,.01),axis=1)
#plt.plot(x,Si(state.params,x))
#plt.plot(x,fi(x))
#plt.show()