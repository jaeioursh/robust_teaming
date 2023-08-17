import numpy as np
import pickle 
import matplotlib.pyplot as plt
import matplotlib as mpl


import pyximport
pyximport.install()
from teaming.cceamtl import *
from mtl import make_env

with open("save/b.pkl","rb") as f:
    arry=pickle.load(f)


env=make_env(1,1)
env.reset()
agent=Evo_MLP(8,2,20)


mp=[[np.nan if a is None else a[1] for a in arr] for arr in arry]
print(mp)
fig1=plt.figure(1)

plt.imshow(mp,cmap=mpl.colormaps["Reds"])
fig2=plt.figure(2)
XX,YY=-1,-1
drawn=False
plot_handle=None

fig2=plt.figure(2)
fig2.clear()
plot_handle, =plt.gca().plot([0],[0],"o")    
print(plot_handle)
pois=env.data["Poi Positions"]

plt.scatter(pois[:,0],pois[:,1])

plt.xlim([-10, 40 ])
plt.ylim([-10, 40 ])

fig2.canvas.draw()


def test(event):
    global XX,YY,drawn,plot_handle
    
    A=[]
    x=int(event.xdata+0.5)
    y=int(event.ydata+0.5)
    if XX==x and YY==y:
        return
    
    
    env.reset()
    XX=x
    YY=y
    print(y,x)
    information=arry[y,x]
    if information is not None:
        params,generation=information
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
        plt.title(np.array2string(S[4:])+"  "+str(r[0])+"   "+str(generation))
        pos=np.array(pos)
        plot_handle.set_xdata(pos[:,0])
        plot_handle.set_ydata(pos[:,1])
        plt.figure(2).canvas.draw()
    
#kind="button_press_event"
kind="motion_notify_event"
fig1.canvas.mpl_connect(kind, test)


plt.show()