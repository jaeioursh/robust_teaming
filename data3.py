#for viewing states in latent space

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from teaming.autoencoder import Autoencoder


def plot3(PERM=1,N=0):
    #PERM=1
    #N=0

    if PERM==1:
        ae=Autoencoder(16)
    else:
        ae=Autoencoder()
    ae.load("save/"+str(PERM)+".mdl")

    if N==0:
        state=np.load("save/"+str(PERM)+".npy")

    if N==1:

        itr=0
        sh=150
        fname="save/baselines/M"+"-".join([str(D) for D in[itr,sh,PERM]])


    if N==2:
        itr=0
        k=5
        AE=1

        fname="save/baselines/N"+"-".join([str(N) for N in[itr,k,AE,PERM]])

    if N==3:
        itr=0
        n_agents=10
        fname="save/baselines/D"+"-".join([str(D) for D in[itr,n_agents,PERM]])


    if N>0:
        fname+=".st.npy"

        state=np.load(fname)
        print(state.shape)



        if PERM==1:
            state=state.reshape((-1,32))
            state=state[:,16:]
            
        else:
            state=state.reshape((-1,8))
            state=state[:,4:]
        
        state=state[:10000]

    titles=["Sampled States", "MASS", "Novelty Search", "DIAYN"]

    xy=ae.feed(state)

    if PERM==0:
        plt.title(titles[N])
    if N==0:
        plt.ylabel("Env. #"+str(PERM+1))

    if 0:
        plt.xlim((0,1))
        plt.ylim((0,1))
        
        plt.scatter(xy[:,0],xy[:,1],s=.5)
    else:
        sh=150
        shape=(sh,sh)
        mp=np.zeros(shape)
        for x,y in xy:
            mp[int(y*sh),int(x*sh)]=1
        plt.imshow(mp,cmap=mpl.colormaps["Reds"])
        plt.xticks([])
        plt.yticks([])


q=0
rows=4
cols=4
plt.rcParams['figure.figsize'] = [10, 10]

for PERM in range(rows):
    for N in range(cols):
        q+=1
        plt.subplot(rows,cols,q)
        plot3(PERM,N)
        plt.gca().set_aspect('equal')
plt.tight_layout()
#plt.subplots_adjust(left=0.005, bottom=0.005, right=.995, top=.995, wspace=0.005, hspace=0.005)
plt.savefig("plots/fig3.png")
plt.show()