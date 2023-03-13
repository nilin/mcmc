import jax.random as rnd
import jax.numpy as np
import pickle
import os


rootkey=rnd.PRNGKey(0)

def nextkeys(n=1):
    global rootkey
    rootkey,*keys=rnd.split(rootkey,n+1)
    return keys

def nextkey():
    return nextkeys()[0]

def save(data,path):
    dirs='/'.join(path.split('/')[:-1])
    os.makedirs(dirs,exist_ok=True)
    with open(path,'wb') as f:
        pickle.dump(data,f)
