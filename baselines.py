import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
import multiprocessing as mp
import time

from mtl import make_env
from teaming.autoencoder import DIAYN,Autoencoder
import pyximport
pyximport.install()
from teaming.cceamtl import *
from train_low import train_map

def diversity(env,itr,n_agents,PERM=0,iterations=4000):
    iterations/=n_agents
    iterations=int(iterations)
    pop_size=32
    if PERM==1:
        agents=[[Evo_MLP(32,2,20) for p in range(pop_size)] for a in range(n_agents)]
    else:
        agents=[[Evo_MLP(8,2,20) for p in range(pop_size)] for a in range(n_agents)]
    if PERM==1:
        desc=DIAYN(n_agents,32)
    else:
        desc=DIAYN(n_agents)
    States=[]
    for i in range(iterations):
        print(i)
        X,Y=[],[]
        POS=[]
        for j in range(n_agents):
            pop=agents[j]
            
            for k in range(pop_size):
                agent=pop[k]
                S=env.reset()[0]
                positions=[]
                for _ in range(30):
                    act=np.array(agent.get_action(S))
                    S, r, done, info = env.step([act])
                    positions.append(np.array(env.data["Agent Positions"]))
                    S=S[0]
                R=desc.forward(S)[j]
                agent.fitness=R
                X.append(S)
                Y.append(j)
                POS.append(positions)
            evolvepop(pop)
        X=np.array(X)
        States.append(X)
        Y=np.array(Y)
        for _ in range(40):
            
            loss=desc.train(X,Y)
            #print(loss)
        if i%10==0:
            fname="save/baselines/D"+"-".join([str(D) for D in[itr,n_agents,PERM]])
            np.save(fname+".st",np.array(States))    
            np.save(fname+".pos",np.array(POS)) 

    fname="save/baselines/D"+"-".join([str(D) for D in[itr,n_agents,PERM]])
    np.save(fname+".st",np.array(States))    
    np.save(fname+".pos",np.array(POS)) 

def neighbors(env,itr,k,AE=0,PERM=0,iterations=451):

    neigh = NearestNeighbors(n_neighbors=k+1)

    pop_size=250
    if AE:
        if PERM==1:
            ae=Autoencoder(16)
        else:
            ae=Autoencoder()
        ae.load("save/"+str(PERM)+".mdl")
    if PERM==1:
        pop=[Evo_MLP(32,2,20) for p in range(pop_size)]
    else:
        pop=[Evo_MLP(8,2,20) for p in range(pop_size)]
    States=[]
    for i in range(iterations):
        print(i)
        X=[]
        POS=[]
        for j in range(pop_size):
            agent=pop[j]
            S=env.reset()[0]
            positions=[]
            for _ in range(30):
                act=np.array(agent.get_action(S))
                S, r, done, info = env.step([act])
                positions.append(np.array(env.data["Agent Positions"]))
                S=S[0]
            X.append(S)
            POS.append(positions)
        States.append(X)
        
      
        if AE:
            if PERM==1:
                x=ae.feed(np.array(X)[:,16:])
            else:
                x=ae.feed(np.array(X)[:,4:])
        else:
            x=X
        neigh.fit(x)
        dist=neigh.kneighbors(x)[0]
        dist=np.array(dist)
        dist=dist[:,1:]
        print(dist.shape)
        #print(dist)
        r=np.mean(dist,axis=1)
        print(np.max(r),r.shape)
        for j in range(pop_size):
            pop[j].fitness=r[j]
        evolvepop(pop)
        if i%10==0:
            fname="save/baselines/N"+"-".join([str(N) for N in[itr,k,AE,PERM]])
            np.save(fname+".st",np.array(States))    
            np.save(fname+".pos",np.array(POS))  


def big_batch():
    for PERM in range(4):
        for itr in range(8,16):
            procs=[]
            
            for k in (5,10,50):
                for AE in [1]:
                    env=make_env(1,PERM=PERM)
                    p=mp.Process(target=neighbors,args=(env,itr,k,AE,PERM))
                    p.start()
                    time.sleep(0.05)
                    procs.append(p)
            for n_agents in (10,50,250):
                env=make_env(1,PERM=PERM)
                p=mp.Process(target=diversity,args=(env,itr,n_agents,PERM))
                p.start()
                time.sleep(0.05)
                procs.append(p)
            
            for sh in [50,150,500]:
                env=make_env(1,PERM=PERM)
                p=mp.Process(target=train_map,args=(env,itr,sh,PERM))
                p.start()
                time.sleep(0.05)
                procs.append(p)
            
            for p in procs:
                p.join()

def small_batch():
    for PERM in range(4):
        procs=[]
        for itr in range(8):
            for sh in [50,150,500]:
                env=make_env(1,PERM=PERM)
                p=mp.Process(target=train_map,args=(env,itr,sh,PERM))
                p.start()
                time.sleep(0.05)
                procs.append(p)
            
        for p in procs:
            p.join()

if __name__ == "__main__":
    if not os.path.exists("save/baselines/"):
        os.makedirs("save/baselines/")
    if 0:
        PERM=0
        env=make_env(1,PERM=PERM)
        #diversity(env,77,10,PERM)
        #neighbors(env,77,5,1,PERM)
        train_map(env,77,50,PERM)
    else:
        big_batch()
        
        