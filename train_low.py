import numpy as np
import matplotlib.pyplot as plt
from random import sample 
import multiprocessing as mp
import time

import pyximport
pyximport.install()
from mtl import make_env
from teaming.gen_agents import run
from teaming.autoencoder import Autoencoder

def sample_states(env,PERM=0):
    S=[]
    for i in range(100000):
        x,t,y=np.random.random(3)
        x=x*40-5
        y=y*40-5
        t=t*np.pi-np.pi

        env.data["Agent Orientations"][0]=[np.sin(t),np.cos(t)]
        env.data["Agent Positions"][0]=[x,y]
        env.data["Observation Function"](env.data)
        if PERM==1:
            s=env.data["Agent Observations"][0][16:]
        else:
            s=env.data["Agent Observations"][0][4:]
        S.append(s)
    
    S=np.array(S)
    np.save("save/"+str(PERM)+".npy",S)

def train_ae(PERM=0):
    S=np.load("save/"+str(PERM)+".npy")
    if PERM==1:
        ae=Autoencoder(16)
    else:
        ae=Autoencoder()
    for i in range(10000):
        s=S[np.random.randint(S.shape[0], size=1000), :]
        err=ae.train(s)
        print(i,err)
        if i%50==49:
            ae.save("save/"+str(PERM)+".mdl")

def view_ae(PERM=0):
    S=np.load("save/"+str(PERM)+".npy")
    if PERM==1:
        ae=Autoencoder(16)
    else:
        ae=Autoencoder()
    ae.load("save/"+str(PERM)+".mdl")
    xy=ae.feed(S)
    plt.scatter(xy[:,0],xy[:,1])
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()

    


def train_map(env,itr,sh,PERM=0):
    
    if PERM==1:
        ae=Autoencoder(16)
    else:
        ae=Autoencoder()
    ae.load("save/"+str(PERM)+".mdl")
    run(env,105000,itr,sh,ae,PERM)


def gen_ae(perm):
    env=make_env(1,1,PERM=perm)
    sample_states(env,perm)
    train_ae(perm)

if __name__=="__main__":
    view_ae(0)
    '''
    if 0:
        procs=[]
        for i in range(4):
            p=mp.Process(target=gen_ae,args=(i,))
            p.start()
            time.sleep(0.05)
            procs.append(p)
        for p in procs:
            p.join()
    else:
        for i in range(4):
            view_ae(i)
    '''
    #train_map(0)

    '''
    
    for itr in range(4):
        sh=150
        env=make_env(1)
        p=mp.Process(target=train_map,args=(itr,sh))
        p.start()
        time.sleep(0.05)
        procs.append(p)
            
                    
    for p in procs:
        p.join()
    '''
