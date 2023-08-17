import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
compact=1
if compact:
    plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
    plt.rcParams['axes.titlepad'] = -14
from teaming import logger
data=[]
err=[]


def shrink(data, cols):
    data=np.array(data[:,:4000])
    print(data.shape)
    return data.reshape(data.shape[0], cols, 4000//cols).mean(axis=2)

AGENTS=5
ROBOTS=4


#i=12
i=4
fname="tests/vary/"+str(AGENTS)+"-"+str(ROBOTS)+"-"+str(i)+"-3.pkl"
log = logger.logger()

#log.load("tests/evo38-5.pkl")
log.load(fname)

r=len(log.pull("reward"))

times=np.array(log.pull("ctime"))
times=times.flatten()
print(times.shape)
times=times.reshape((r,-1))
probs=np.array([np.bincount(t,minlength=30).astype(float)/r for t in times])
print(probs.shape)
probs=shrink(probs.T,40)
scale=np.sum(probs,axis=0)[0]
probs/=scale
#probs=np.log(probs)
#probs[np.isinf(probs)]=-8
plt.imshow(probs,interpolation="none",aspect='auto',cmap=matplotlib.colormaps["Reds"])
plt.xlabel("Generation")
plt.ylabel("Index Along State Trajectory")
plt.colorbar(label="Probability of Being Max Valued")
plt.tight_layout()

tx=len(plt.gca().get_xticklabels())
print(tx)
x=[i*4 for i in range(tx)]
labels=[str(i*100) for i in x]
print(x,labels)
plt.xticks(x, labels)
plt.show()
