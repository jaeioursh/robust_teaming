#for coverage curve
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('tableau-colorblind10')

from teaming.autoencoder import Autoencoder
'''
scp -J cookjos@access.engr.oregonstate.edu cookjos@graf200-16.engr.oregonstate.edu:robust_teaming/save/baselines/* save/baselines/
'''
def loads(fname,ae,shape,PERM):
    state=np.load(fname)
    print(state.shape)
    if PERM==1:
        if len(state.shape)==3:
            state=state.reshape((state.shape[0]*state.shape[1],32))
        state=state[:,16:]
        
    else:
        if len(state.shape)==3:
            state=state.reshape((state.shape[0]*state.shape[1],8))
        state=state[:,4:]
    print(state.shape)
    state=state[:100000]
    states=np.split(state,10)
    
    print("Net in")
    xy=np.vstack([ae.feed(s) for s in states])
    print("Net out")
    print(xy.shape)
    arry=np.zeros(shape,dtype=object)
    arry[:]=None
    count=0
    counts=[]
    for i in range(len(state)):
        x=int(xy[i,0]*shape[0])
        y=int(xy[i,1]*shape[1])
        if arry[x,y] is None:
            count+=1
            arry[x,y]=count
        counts.append(count)
    return counts,arry



def plot4(PERM,sh=50,view=False):
    #PERM=1
    TRIALS=8
    if PERM==1:
        ae=Autoencoder(16)
    else:
        ae=Autoencoder()
    ae.load("save/"+str(PERM)+".mdl")
    #sh=50
    shape=(sh,sh)

    data=[]
    for k in (5,10,50):
        for AE in [1]:
            d=[]
            for itr in range(TRIALS):
                fname="save/baselines/N"+"-".join([str(N) for N in[itr,k,AE,PERM]])+".st.npy"
                print(fname)
                d.append( loads(fname,ae,shape,PERM)[0] )
            data.append([d,"Diversity-"+str(k)])#+"ae"*AE])

    for n_agents in (10,50,250):
        d=[]
        for itr in range(TRIALS):
            fname="save/baselines/D"+"-".join([str(D) for D in[itr,n_agents,PERM]])+".st.npy"
            print(fname)
            d.append( loads(fname,ae,shape,PERM)[0] )
        data.append([d,"DIAYN-"+str(n_agents)])

    d=[]
    for itr in range(TRIALS):
        fname="save/baselines/M"+"-".join([str(D) for D in[itr,sh,PERM]])+".st.npy"
        print(fname)
        d.append( loads(fname,ae,shape,PERM)[0] )
            

    data=sorted(data,key = lambda Q:-np.mean([q[-1] for q in Q[0]]))
    data=[[d,"OURS"]]+data
    for d,tag in data:
        T=np.mean(d,axis=0)
        X=np.arange(len(T))
        std=np.std(d,axis=0)/np.sqrt(8)
        plt.plot(X,T,label=tag)
        plt.fill_between(X,T-std,T+std,alpha=0.35, label='_nolegend_')
    plt.legend()
    plt.title("Resolution: " +str(sh)+", Env. Version "+str(PERM))
    plt.xlabel("Episodes")
    plt.ylabel("Coverage")
    plt.savefig()
    if view:
        plt.show()

def build():
    import cv2
    image=[]
    for sh in [50,150,500]:
        ims=[]
        for PERM in range(4):
            fname = "plots/fig4-"+str(PERM)+"-"+str(sh)+".png"
            im = cv2.imread(fname)
            ims.append(im)
        image.append(np.hstack(ims))
    image=np.vstack(image)
    cv2.imwrite("plots/fig4.png",image)
if __name__ == "__main__":
    
    if 1:
        build()
    else:
        import multiprocessing as mp
        import time
        procs=[]
        for sh in [50,150,500]:
            for PERM in range(4):
                p=mp.Process(target=plot4,args=(PERM,sh))
                p.start()
                time.sleep(2)
                procs.append(p)
            
        for p in procs:
            p.join()