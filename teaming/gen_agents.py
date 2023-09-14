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

def pick(arry,MT):
    ij=MT.nonzero()

    idx=np.random.randint(0,len(ij[0]))
    return arry[ij[0][idx],ij[1][idx]]
    #return np.random.choice(vals,1)[0]


def run(env,iters,itr,sh,ae=None,PERM=0):
    n=0
    shape=(sh,sh)
    arry=gen_dict(shape)
    MT=np.zeros(shape)
    if PERM==1:
        agent=Evo_MLP(32,2,20)
    else:
        agent=Evo_MLP(8,2,20)
    STATES=[]
    POS=[]
    for i in range(iters):
       
        S=env.reset()[0]
        if i>0:
            params,r,old_idx,hits=pick(arry,MT)
            params=[np.copy(np.array(p)) for p in params]
            agent.__setstate__(params)
            agent.mutate()
        
        positions=[]
        for j in range(30):
            act=np.array(agent.get_action(S))
            S, r, done, info = env.step([act])
            positions.append(np.array(env.data["Agent Positions"]))
            S=S[0]
        STATES.append(S)
        POS.append(positions)
        g=0.0
        r=r[0]
        
        if ae is None:
            idx=pol2idx(S,shape)
        else:
            if PERM==1:
                ix=ae.feed(S[16:])
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
            MT[idx]=1
        else:
            if idx!=old_idx:
                arry[idx][-1]+=1



        
        

        if i%1000==0:
            
            print(i,n)
            fname="save/baselines/M"+"-".join([str(D) for D in[itr,sh,PERM]])
            with open(fname+".pkl","wb") as f:
                pickle.dump( arry,f)
            np.save(fname+".st",np.array(STATES))  
            np.save(fname+".pos",np.array(POS))  

