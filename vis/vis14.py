#alignment/vis
from mtl import make_env

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import pickle

import code.reward_2 as reward
from teaming import logger
from teaming.learnmtl import Net

q=3
i=4
AGENTS=5
ROBOTS=4

fname="tests/very/"+str(AGENTS)+"-"+str(ROBOTS)+"-"+str(i)+"-"+str(q)+".pkl"

log = logger.logger()
log.load(fname)
env=make_env(ROBOTS)
pos=log.pull("position")
teams=np.array(log.pull("types")[0])
INDEX=1


net=Net(20)
net.model.load_state_dict(torch.load(fname+".mdl")[INDEX])

#env.reset()
def eval(x,y,t,i,TEAM,r=0):
    test=pos[i][TEAM].copy()
    test[teams[TEAM]==INDEX,:]=[x,y]
    env.data["Agent Orientations"][teams[TEAM]==INDEX,:]=[np.sin(t),np.cos(t)]
    env.data["Agent Positions"]=test
    env.data["Agent Position History"]=np.array([env.data["Agent Positions"]])
    env.data["Steps"]=0
    env.data["Observation Function"](env.data)
    z=env.data["Agent Observations"]
    s=z[teams[TEAM]==INDEX]


    #reward.assignGlobalReward(env.data)
    #reward.assignDifferenceReward(env.data)
    if r:
        env.data["Reward Function"](env.data)
        g=env.data["Global Reward"]
    else:
        g=0
    #print(env.data["Agent Orientations"])
    return net.feed(s)[0,0],g
def eval2(steps,TEAM):
    x=np.linspace(-5,35,steps)
    y=x.copy()

    zz=np.zeros((len(x),len(y)))
    zz2=zz.copy()
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(20):
                t=np.random.random()*2*np.pi
                q=-1
                #q=np.random.randint(0,30)
                ghat,g=eval(x[i],y[j],t,q,TEAM,1)
                zz[j,i]+=ghat
                zz2[j,i]+=g
    return zz,zz2

ts=[0,1,2,4]

VALS=[0.8,1.0,0.6,0.3,0.2,0.1]
poi=log.pull("poi")[0]
txt=[str(i) for i in VALS]

CMAP=matplotlib.colormaps["plasma"]

if 0:
    data=[]
    for i in range(len(ts)):
        zz,zz2=eval2(100,ts[i])
        data.append([zz,zz2])
    with open("tests/test.data", 'wb') as f:
        pickle.dump(data, f)

with open("tests/test.data", 'rb') as f:
    data=pickle.load(f)

for i in range(len(ts)):
    zz,zz2=data[i]

    ext=[-5,35,-5,35]
    plt.subplot(1,4,i+1)
    plt.title("Team "+str(ts[i]))#+", Agent "+str(INDEX))
    ax=plt.gca()
    if 0:
        zz[zz<0.3]=0.3
        im=ax.imshow(np.flipud(zz)/20,extent=ext,cmap=CMAP)
        lbl="Critic Value"
        
    else:
        im=ax.imshow(np.flipud(zz2)/20,extent=ext,cmap=CMAP)
        lbl="Team Value"

    for i in range(len(txt)):
        plt.text(poi[i,0]-3,poi[i,1]+2,txt[i],c='#ffffff')
    plt.scatter(poi[:,0],poi[:,1],s=50,c='#ffffff',marker="v",zorder=10000)
    plt.xlim([-5,35])
    plt.ylim([-5,35])
    plt.axis('off')
plt.subplots_adjust(left=0.05, bottom=0.05, right=.87, top=.95, wspace=0.05, hspace=0.05)

cax = plt.gcf().add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
plt.colorbar(im, label=lbl, cax=cax)
plt.show()