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
AGENTS=5
ROBOTS=4
ROWS=10
COLS=10
N=100
fname="save/testing/data4-25-500-3.pkl"

log = logger.logger()
#log.load("tests/evo38-5.pkl")
log.load(fname)
p=log.pull("position")
tst=log.pull("test")


tst=np.array(tst)
Tst=np.average(tst,axis=1)
tst=tst[-1]



poi=log.pull("poi")[0]



nagents=4
pos=p[0]

print(np.array(pos).shape)
summ=0

for idx in range(N):
    plt.subplot(ROWS,COLS,idx+1)#range(50):
    #for idx in [40]:#range(50):
    #plt.ion()
    #plt.clf()
    #plt.subplot(1,2,1)
    VALS=[0.8,1.0,0.6,0.3,0.2,0.1]
 
    txt=[str(i) for i in VALS]
    vals=np.array(VALS)*0+1000
    
    
  

    for i in range(nagents):
        data=[]
        for j in range(31):
            #print(np.array(pos).shape)
            p=pos[idx][j][0][i]
            data.append(p)
        data=np.array(data).T
        x,y=data
        if i==nagents-1:
            tt=1
        else:
            tt=0
        mkr=[".",",","*","v","^","<",">","1","2","3","4","8"][tt]
        clr=["b","g","c","m","y","k","r","b","g","c","m","y"][tt]
        #mkr='*'
        #clr="k"
        plt.plot(x,y,color=clr,marker=mkr,linewidth=1.0)
    
    
        
    #print(lgnd)

    #plt.legend(lgnd)
    if compact and 0:
        plt.xticks([])
        plt.yticks([])
    for i in range(len(txt)):
        plt.text(poi[i,0]+1,poi[i,1]+1,txt[i])

    plt.scatter(poi[:,0],poi[:,1],s=10,c='#0000ff',marker="v",zorder=10000)
    plt.xlim([-5,35])
    plt.ylim([-5,35])
    plt.title(str(idx)+':'+str(np.round(tst[idx],2)))

    summ+=tst[idx]
    #plt.title("Team "+str(idx)+" Score: "+str(round(tst[idx],3)))
    #if tst[idx]<0.6:
    #    continue
    #plt.subplot(1,2,2)
    #plt.plot(Tst)
    #plt.axes().set_aspect('equal', 'datalim')
    #plt.pause(1.0)
print(summ)
if compact:
    plt.subplots_adjust(left=0.005, bottom=0.005, right=.995, top=.995, wspace=0.005, hspace=0.005)
plt.show()