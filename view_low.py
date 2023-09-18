import numpy as np
import pickle 
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import time

import pyximport
pyximport.install()
from teaming.cceamtl import *
from mtl import make_env
itr=0
sh=150
PERM=0
fname="save/baselines/M"+"-".join([str(D) for D in[itr,sh,PERM]])+".pkl"
print(fname)
with open(fname,"rb") as f:
    arry=pickle.load(f)


env=make_env(1,1)
env.reset()
agent=Evo_MLP(8,2,20)


mp=[[np.nan if a is None else a[1] for a in arr] for arr in arry]
#print(mp)
fig1=plt.figure(1)

plt.imshow(mp,cmap=mpl.colormaps["Reds"])

mp=[[np.nan if a is None else 1./(a[3]+200) for a in arr] for arr in arry]
#print(mp)
fig1_5=plt.figure(2)

plt.imshow(mp,cmap=mpl.colormaps["Reds"])

fig2=plt.figure(3)
plot_handle=None
XX,YY=-1,-1
drawn=False
plot_handle=None

fig2=plt.figure(3)
fig2.clear()
plot_handle, =plt.gca().plot([0],[0],"o")    
print(plot_handle)
pois=env.data["Poi Positions"]

plt.scatter(pois[:,0],pois[:,1])

plt.xlim([-10, 40 ])
plt.ylim([-10, 40 ])

fig2.canvas.draw()
T=time()

def test(event):
    global XX,YY,drawn,plot_handle,T
    if time()-T<1.0/5.0:
        return
    else:
        T=time()

    A=[]
    x=int(event.xdata+0.5)
    y=int(event.ydata+0.5)
    if XX==x and YY==y:
        return
    
    
    
    XX=x
    YY=y
    print(y,x)
    information=arry[y,x]
    if information is not None:
        env.reset()
        params=information[0]
        params=[np.copy(np.array(p)) for p in params]
        agent.__setstate__(params)

        S=env.reset()[0]
        #print(S)
        pos=[]
        for j in range(30):
            act=np.array(agent.get_action(S))
            S, r, done, info = env.step([act])
            S=S[0]
            xy=env.data["Agent Positions"][0]
            pos.append(xy.copy())
        S=np.round(S,4)
        plt.title(np.array2string(S[4:]))
        pos=np.array(pos)
        plot_handle.set_xdata(pos[:,0])
        plot_handle.set_ydata(pos[:,1])
        plt.gcf().canvas.draw()
    
#kind="button_press_event"
kind="motion_notify_event"
fig1.canvas.mpl_connect(kind, test)


plt.show()