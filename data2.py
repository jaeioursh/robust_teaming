#for viewing paths
import numpy as np
import matplotlib.pyplot as plt

novelty=0

if novelty:
    itr=0
    k=5
    AE=0

    fname="save/baselines/N"+"-".join([str(N) for N in[itr,k,AE]])

else:
    itr=0
    n_agents=250
    fname="save/baselines/D"+"-".join([str(D) for D in[itr,n_agents]])


fname+=".pos.npy"

pos=np.load(fname)
print(pos.shape)
for p in pos:
    
    p=np.array([p0[0] for p0 in p])
    plt.plot(p[:,0],p[:,1])
plt.xlim((-5,35))
plt.ylim((-5,35))
plt.show()