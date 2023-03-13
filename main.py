import jax.numpy as jnp
import jax.random as rnd
import sys
import optax
from flax.training.train_state import TrainState
from session import nextkey
import mcmc, session
import energy
import tests



if 'm' in sys.argv:
    mode='mcmc'
elif 'c' in sys.argv:
    mode='correlated'
elif 'n' in sys.argv:
    mode='no_mc'
else:
    print('\nchoose mode m=MCMC, c=correlated, n=NO_MC\n')
    quit()


model=tests.Model()
slog_Si=model.apply
Si_=energy.noslog(slog_Si)
Si=lambda P,X: Si_(P,X)*jnp.squeeze(jnp.abs(X)<1)
P_Si=lambda P,X: Si(P,X)**2

X=rnd.uniform(nextkey(),(1000,1),minval=-1,maxval=1)
Y=rnd.uniform(nextkey(),(1000,1),minval=-1,maxval=1)
params0=model.init(nextkey(),X)

fi=lambda x: jnp.squeeze((jnp.sin(5*x)+1)*(jnp.abs(x)<1))
P_Fi=lambda _,x: fi(x)**2

state=TrainState.create(apply_fn=slog_Si,params=params0,tx=optax.sgd(.1,.2))
vals=[]
nburn=1000
proposal=mcmc.gaussianproposal(.5)

match mode:
    case 'mcmc':
        E_value_and_grad=energy.MCMC_value_and_grad(slog_Si,fi=fi)

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

        session.save(vals,'outputs/mcmc')


    case 'correlated':
        E_value_and_grad=energy.correlated_value_and_grad(slog_Si,fi)

        fi_sampler=mcmc.Metropolis(P=P_Fi,proposal=proposal,walkers=Y)
        Y=fi_sampler.sample(None,steps=nburn)

        for i in range(1000):
            Y=fi_sampler.sample(None)

            loss,grads=E_value_and_grad(state.params,X,Y)
            state=state.apply_gradients(grads=grads)
            vals.append(-loss)
            print(-loss)

        session.save(vals,'outputs/correlated')


    case 'no_mc':
        E_value_and_grad=energy.NO_MC_value_and_grad(slog_Si,fi)

        for i in range(1000):
            loss,grads=E_value_and_grad(state.params,X,Y)
            state=state.apply_gradients(grads=grads)
            vals.append(-loss)
            print(-loss)

        session.save(vals,'outputs/no_mc')

        import matplotlib.pyplot as plt
        x=jnp.expand_dims(jnp.arange(-2,2,.01),axis=1)
        plt.plot(x,Si(state.params,x))
        plt.plot(x,fi(x))
        plt.show()