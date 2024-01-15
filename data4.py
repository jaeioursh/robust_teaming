#for coverage curve
import numpy as np
import matplotlib.pyplot as plt
import pickle

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



def pre_plot(PERM,sh=50,view=False):
    #PERM=1
    TRIALS=24
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
    data=[[d,"MASS"]]+data

    with open("plots/fig4-"+str(PERM)+"-"+str(sh)+".pkl","wb") as f:
        pickle.dump(data,f)

def plot4(PERM,sh=50):
    with open("plots/fig4-"+str(PERM)+"-"+str(sh)+".pkl","rb") as f:
        print("loading")
        data=pickle.load(f)
        print("loaded")
    TRIALS=len(data[0][0])
    for d,tag in data:
        res=1000
        d=np.array(d)
        d=d[:,::res]
        T=np.mean(d,axis=0)
        
        X=np.arange(len(T))*res
        std=np.std(d,axis=0)/np.sqrt(TRIALS)
        
        plt.plot(X,T,label=tag)
        plt.fill_between(X,T-std,T+std,alpha=0.35, label='_nolegend_')
    if PERM==3 and sh==50:
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    if sh==50:
        plt.title("Env. Version "+str(PERM+1))
    if sh==500:
        plt.xlabel("Episodes")
    if PERM==0:
        plt.ylabel("Resolution: " +str(sh)+"\n Coverage")
    plt.grid()
    #plt.savefig("plots/fig4-"+str(PERM)+"-"+str(sh)+".png")
   
    

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
    MAKE=0
    if 0:
        build()
    else:
        import multiprocessing as mp
        import time
        
        nplot=0
        for sh in [50,150,500]:
            procs=[]
            for PERM in range(4):
                nplot+=1
                print(nplot)
                if MAKE:
                    p=mp.Process(target=pre_plot,args=(PERM,sh))
                    p.start()
                    time.sleep(2)
                    procs.append(p)
                else:
                    plt.subplot(3,4,nplot)
                    plot4(PERM,sh)
            if MAKE:
                for p in procs:
                    p.join()
        if not MAKE:
            plt.tight_layout()
            plt.show()