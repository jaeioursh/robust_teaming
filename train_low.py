import numpy as np
import matplotlib.pyplot as plt
from random import sample 

import pyximport
pyximport.install()
from mtl import make_env
from teaming.gen_agents import run
from teaming.autoencoder import Autoencoder

def sample_states():
    env = make_env(1)
    S=[]
    for i in range(100000):
        x,t,y=np.random.random(3)
        x=x*40-5
        y=y*40-5
        t=t*np.pi-np.pi

        env.data["Agent Orientations"][0]=[np.sin(t),np.cos(t)]
        env.data["Agent Positions"][0]=[x,y]
        env.data["Observation Function"](env.data)
        s=env.data["Agent Observations"][0][4:]
        S.append(s)
    
    S=np.array(S)
    np.save("save/a.npy",S)

def train_ae():
    S=np.load("save/a.npy")
    ae=Autoencoder()
    for i in range(10000):
        s=S[np.random.randint(S.shape[0], size=1000), :]
        err=ae.train(s)
        print(i,err)
        if i%50==49:
            ae.save("save/a.mdl")

def view_ae():
    S=np.load("save/a.npy")
    ae=Autoencoder()
    ae.load("save/a.mdl")
    xy=ae.feed(S)
    plt.scatter(xy[:,0],xy[:,1])
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()

    


def train_map():
    env = make_env(1,1)
    ae=Autoencoder()
    ae.load("save/a.mdl")
    run(env,1000000,ae)


if __name__=="__main__":
    #sample_states()
    #train_ae()
    #view_ae()
    train_map()
