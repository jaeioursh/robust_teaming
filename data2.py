#for viewing paths
import numpy as np
import matplotlib.pyplot as plt

def plot2(PERM=1,N=0):
    #PERM=1
    #N=0
    itr=0

    if N==1:

        sh=500
        fname="save/baselines/M"+"-".join([str(D) for D in[itr,sh,PERM]])


    if N==2:
        k=5
        AE=1

        fname="save/baselines/N"+"-".join([str(N) for N in[itr,k,AE,PERM]])

    if N==3:
        n_agents=10
        fname="save/baselines/D"+"-".join([str(D) for D in[itr,n_agents,PERM]])

    titles=["Sampled States", "MASS", "Novelty Search", "DIAYN"]
    if PERM==0:
        plt.title(titles[N])
    if N==1:
        plt.ylabel("Env. #"+str(PERM+1))

    fname+=".pos.npy"

    pos=np.load(fname)
    print(pos.shape)

    if len(pos)>1000:

        idx=np.arange(len(pos))
        np.random.shuffle(idx)
        idx=idx[:500]
        pos=pos[idx,:,:,:]
        print(len(pos))
    for p in pos:
        
        p=np.array([p0[0] for p0 in p])
        plt.plot(p[:,0],p[:,1],"k-",linewidth=0.1)
    
    plt.xticks([])
    plt.yticks([])
    plt.xlim((-5,35))
    plt.ylim((-5,35))



q=0
rows=4
cols=3
plt.rcParams['figure.figsize'] = [8, 12]

for PERM in range(rows):
    for N in range(1,cols+1):
        q+=1
        print(q)
        plt.subplot(rows,cols,q)
        plot2(PERM,N)
        plt.gca().set_aspect('equal')
plt.tight_layout()
#plt.subplots_adjust(left=0.005, bottom=0.005, right=.995, top=.995, wspace=0.005, hspace=0.005)
plt.savefig("plots/fig2.png")
plt.show()