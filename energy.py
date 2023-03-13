import jax, jax.numpy as jnp
import flax
from functools import partial


clip=lambda x:jnp.clip(x,a_min=-100,a_max=100)

def E_value_and_grad(slog_Si,fi):
    Si=noslog(slog_Si)

    def value_and_grad(P,X,Y):
        si=partial(Si,P)
        RX=fi(X)/si(X)
        RY=si(Y)/fi(Y)
        E=jnp.average(RX*RY)
        grad=jax.grad(lambda p: -jnp.average( clip( (RX-jnp.average(RX))*slog_Si(p,X)[1] ))*jnp.average(RY))
        return E,grad(P)
    
    return jax.jit(value_and_grad)



def nomcmc_value_and_grad(slog_Si,fi):
    Si=noslog(slog_Si)

    def loss(P,X,Y):
        #sifi=jnp.sqrt(jnp.average( clip((Si(P,Y)**2)/(fiY**2)) ))
        sifi=jnp.sqrt(jnp.sum(Si(P,X)**2)/jnp.sum(fi(X)**2))
        return -jnp.average( clip(Si(P,Y)/fi(Y)) )/sifi

    return jax.jit(jax.value_and_grad(loss))



    
def noslog(slog_f):
    def f(P,X):
        s,l=slog_f(P,X)
        return s*jnp.exp(l)
    return f
