#for viewing states in latent space

import numpy as np
import matplotlib.pyplot as plt

from teaming.autoencoder import Autoencoder

novelty=0

if novelty:
    itr=0
    k=5
    AE=1

    fname="save/baselines/N"+"-".join([str(N) for N in[itr,k,AE]])

else:
    itr=0
    n_agents=250
    fname="save/baselines/D"+"-".join([str(D) for D in[itr,n_agents]])


fname+=".st.npy"

state=np.load(fname)
print(state.shape)
state=state.reshape((state.shape[0]*state.shape[1],8))
state=state[:,4:]
state=state[:100000]
ae=Autoencoder()
ae.load("save/a.mdl")
xy=ae.feed(state)

plt.xlim((0,1))
plt.ylim((0,1))
plt.scatter(xy[:,0],xy[:,1])
plt.show()