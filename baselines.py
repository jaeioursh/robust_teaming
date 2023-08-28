import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
import multiprocessing as mp
from time import time

from mtl import make_env
from teaming.autoencoder import DIAYN,Autoencoder
import pyximport
pyximport.install()
from teaming.cceamtl import *

def diversity(env,itr,n_agents,iterations=1000):
    
    pop_size=32
    agents=[[Evo_MLP(8,2,20) for p in range(pop_size)] for a in range(n_agents)]
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
            fname="save/baselines/D"+"-".join([str(D) for D in[itr,n_agents]])
            np.save(fname+".st",np.array(States))    
            np.save(fname+".pos",np.array(POS)) 

def neighbors(env,itr,k,AE=0,iterations=2000):

    neigh = NearestNeighbors(n_neighbors=k+1)

    pop_size=250
    if AE:
        ae=Autoencoder()
        ae.load("save/a.mdl")
    
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
            x=ae.feed(np.array(X)[:,4:])
        else:
            x=X
        neigh.fit(x)
        dist=neigh.kneighbors(x)[0]
        dist=np.array(dist)
        dist=dist[:,1:]
        r=np.mean(dist,axis=1)

        for j in range(pop_size):
            pop[j].fitness=r[j]
        evolvepop(pop)
        if i%10==0:
            fname="save/baselines/N"+"-".join([str(N) for N in[itr,k,AE]])
            np.save(fname+".st",np.array(States))    
            np.save(fname+".pos",np.array(POS))  


if __name__ == "__main__":
    if not os.path.exists("save/baselines/"):
        os.makedirs("save/baselines/")
    if 1:
        env=make_env(1)
        diversity(env,0,10)
        #neighbors(env,0,5,0)
    else:
        procs=[]
        
        for itr in range(4):
            for k in (5,10,50):
                for AE in [0,1]:
                    env=make_env(1)
                    p=mp.Process(target=neighbors,args=(env,itr,k,AE))
                    p.start()
                    time.sleep(0.05)
                    procs.append(p)
            for n_agents in (10,50,250):
                env=make_env(1)
                p=mp.Process(target=diversity,args=(env,itr,n_agents))
                p.start()
                time.sleep(0.05)
                procs.append(p)
                    
        for p in procs:
            p.join()