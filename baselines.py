import numpy as np
import os
from sklearn.neighbors import NearestNeighbors


from mtl import make_env
from teaming.autoencoder import DIAYN,Autoencoder
import pyximport
pyximport.install()
from teaming.cceamtl import *

def diversity(itr,n_agents,iterations=1000):
    env=make_env(1)
    pop_size=32
    agents=[[Evo_MLP(8,2,20) for p in range(pop_size)] for a in range(n_agents)]
    desc=DIAYN(n_agents)
    States=[]
    for i in range(iterations):
        print(i)
        X,Y=[],[]
        for j in range(n_agents):
            pop=agents[j]
            
            for k in range(pop_size):
                agent=pop[k]
                S=env.reset()[0]
                for _ in range(30):
                    act=np.array(agent.get_action(S))
                    S, r, done, info = env.step([act])
                    S=S[0]
                R=desc.forward(S)[j]
                agent.fitness=R
                X.append(S)
                Y.append(j)
            evolvepop(pop)
        X=np.array(X)
        States.append(X)
        Y=np.array(Y)
        for i in range(40):
            print(desc.train(X,Y))
        if i%10==0:
            fname="save/baselines/D"+"-".join([str(D) for D in[itr,n_agents]])
            np.save(fname,np.array(States))

def neighbors(itr,k,AE=0,iterations=1000):

    neigh = NearestNeighbors(n_neighbors=k+1)
    env=make_env(1)
    pop_size=250
    if AE:
        ae=Autoencoder()
        ae.load("save/a.mdl")
    
    pop=[Evo_MLP(8,2,20) for p in range(pop_size)]
    States=[]
    for i in range(iterations):
        X=[]
        for j in range(pop_size):
            agent=pop[j]
            S=env.reset()[0]
            for _ in range(30):
                act=np.array(agent.get_action(S))
                S, r, done, info = env.step([act])
                S=S[0]
            X.append(S)
        States.append(X)
        print(X)
        if AE:
            x=ae.feed(np.array(X)[:,4:])
        else:
            x=X
        neigh.fit(x)
        dist=neigh.kneighbors(x)[0]
        dist=np.array(dist)
        dist=dist[:,1:]
        r=np.mean(dist,axis=1)
        print(dist.shape,r.shape)
        for j in range(pop_size):
            pop[j].fitness=r[j]
        evolvepop(pop)
        if i%10==0:
            fname="save/baselines/N"+"-".join([str(N) for N in[itr,k,AE]])
            np.save(fname,np.array(States))    


if __name__ == "__main__":
    if not os.path.exists("save/baselines/"):
        os.makedirs("save/baselines/")
    #diversity(0,10)
    neighbors(0,5,1)