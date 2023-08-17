import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def plot(agents,team_size,generation, trial,ROWS=1,COLS=1):
    compact=1
    if compact:
        plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
        plt.rcParams['axes.titlepad'] = -14
    from teaming import logger
    data=[]
    err=[]
    AGENTS=agents
    ROBOTS=team_size
    

 

    q=3
    fname="tests/"+str(agents)+"-"+str(team_size)+"-"+str(trial)+"-"+str(q)+"/data"

    log = logger.logger()
    log.load(fname+".pkl")
    p=log.pull("position")
    t=log.pull("types")
    tst=log.pull("test")


    tst=np.array(tst)
    Tst=np.average(tst,axis=1)
    tst=tst[-1]



    poi=log.pull("poi")[0]



    nagents=len(t[0][0])
    pos=p

    N=len(t[0])
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
        
        
        typ=t[0][idx]

        for i in range(nagents):
            data=[]
            for j in range(len(pos[generation][idx])):
                #print(np.array(pos).shape)
                p=pos[generation][idx][j][0][i]
                data.append(p)
            data=np.array(data).T
            x,y=data
            tt=typ[i]
            mkr=[".",",","*","v","^","<",">","1","2","3","4","8"][tt]
            clr=["b","g","c","m","y","k","r","b","g","c","m","y"][tt]
            #mkr='*'
            #clr="k"
            plt.plot(x,y,color=clr,marker=mkr,linewidth=1.0)
        if compact and 0:
            lgnd=[str(i) for i in typ]
        else:
            lgnd=["Agent "+str(i) for i in typ]
            
        #print(lgnd)

        plt.legend(lgnd)
        if compact and 0:
            plt.xticks([])
            plt.yticks([])
        for i in range(len(txt)):
            plt.text(poi[i,0]+1,poi[i,1]+1,txt[i])

        plt.scatter(poi[:,0],poi[:,1],s=100,c='#0000ff',marker="v",zorder=10000)
        plt.xlim([-5,35])
        plt.ylim([-5,35])
        plt.title("Team "+str(idx)+' score:'+str(tst[idx]))

        pois=env.data["Poi Positions"]
        
        plt.scatter(pois[:,0],pois[:,1])

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
        plt.subplots_adjust(left=0.05, bottom=0.05, right=.95, top=.95, wspace=0.05, hspace=0.05)
    plt.show()

agents=5
team_size=4
generation=3000
trial=0
ROWS=1
COLS=5
plot(agents,team_size,generation, trial,ROWS,COLS)