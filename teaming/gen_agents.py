import numpy as np
import pickle 

from .cceamtl import *


def gen_dict(shape):
    arry=np.zeros(shape,dtype=object)
    arry[:]=None
    return arry

def pol2idx(info,ix):
    scale=500
    #print(info)
    i1,i2=info[7]+info[6],info[5]+info[4]
    i1,i2=min(int(i1*scale),ix[0]-1),min(int(i2*scale),ix[1]-1)
    return (i1,i2)

def pick(arry):
    vals=arry[arry!=None]
    return np.random.choice(vals,1)[0]


def run(env,iters,ae=None):
    n=0
    shape=(50,50)
    arry=gen_dict(shape)
    agent=Evo_MLP(8,2,20)
    for i in range(iters):
       
        S=env.reset()[0]
        if i>0:
            params,r,old_idx,hits=pick(arry)
            params=[np.copy(np.array(p)) for p in params]
            agent.__setstate__(params)
            agent.mutate()
            
        for j in range(30):
            act=np.array(agent.get_action(S))
            S, r, done, info = env.step([act])
            S=S[0]
        g=0.0
        r=r[0]
        
        if ae is None:
            idx=pol2idx(S,shape)
        else:
            ix=ae.feed(S[4:])
            idx=[]
            for q in range(len(shape)):
                idx.append(int(ix[q]*shape[q]))
            idx=tuple(idx)

        info=arry[idx]
        if info is None:
            n+=1
            params=[np.copy(np.array(p)) for p in agent.__getstate__()]
            arry[idx]=[params,r,idx,0]
        else:
            if idx!=old_idx:
                arry[idx][-1]+=1



        
        

        if i%1000==0:
            print(i,n)
            with open("save/a.pkl","wb") as f:
                pickle.dump( arry,f)

