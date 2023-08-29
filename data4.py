#for coverage curve
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('tableau-colorblind10')

from teaming.autoencoder import Autoencoder

#scp -J cookjos@access.engr.oregonstate.edu cookjos@graf200-16.engr.oregonstate.edu:robust_teaming/save/baselines/* save/baselines/
def loads(fname,ae,shape):
    state=np.load(fname)
    print(state.shape)
    state=state.reshape((state.shape[0]*state.shape[1],8))
    state=state[:,4:]
    state=state[:10000]
    states=np.split(state,1000)
    
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





ae=Autoencoder()
ae.load("save/a.mdl")
shape=(50,50)

data=[]
for k in (5,10,50):
    for AE in [0,1]:
        d=[]
        for itr in range(4):
            fname="save/baselines/N"+"-".join([str(N) for N in[itr,k,AE]])+".st.npy"
            d.append( loads(fname,ae,shape)[0] )
        data.append([d,"Diversity"+str(k)+"ae"*AE])
for n_agents in (10,50,250):
    d=[]
    for itr in range(4):
        fname="save/baselines/D"+"-".join([str(D) for D in[itr,n_agents]])+".st.npy"
        d.append( loads(fname,ae,shape)[0] )
    data.append([d,"DIAYN"+str(n_agents)])
        

for d,tag in data:
    T=np.mean(d,axis=0)
    X=np.arange(len(T))
    std=np.std(d,axis=0)/2
    plt.plot(X,T,label=tag)
    plt.fill_between(X,T-std,T+std,alpha=0.35, label='_nolegend_')
plt.legend()
plt.show()



    




plt.plot()
plt.show()