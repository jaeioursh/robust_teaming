from teaming.gen_agents import run
from teaming.autoencoder import Autoencoder
import numpy as np
from mtl import make_env
import multiprocessing as mp
import time

def train_both(env,itr=-1,sh=50,iters=1000):
    S=[]
    P=[]
    
    
    for idx in range(105000//iters):
        print(idx)
        ae=Autoencoder()
        Snp=np.array(S)
        if len(S)>200:
            for i in range(100):
                s=Snp[np.random.randint(Snp.shape[0], size=200), :4]
                err=ae.train(s)
            
        ST,POS=run(env,iters,itr,sh,ae=None,PERM=0,saved=False)
        S+=ST
        P+=POS

    fname="save/baselines/B"+"-".join([str(D) for D in[itr,sh]])
    #with open(fname+".pkl","wb") as f:
    #    pickle.dump( arry,f)
    np.save(fname+".st",np.array(S))  
    np.save(fname+".pos",np.array(P)) 


if __name__=="__main__":
    procs=[]
    for i in range(16):
        env=make_env(1,PERM=0)
        p=mp.Process(target=train_both,args=(env,i,50,1000))
        p.start()
        time.sleep(0.05)
        procs.append(p)
    
    for p in procs:
        p.join()