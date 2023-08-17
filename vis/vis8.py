from urllib.request import CacheFTPHandler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#from math import comb
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.style.use('tableau-colorblind10')
#matplotlib.rcParams['text.usetex'] = True
from teaming import logger
DEBUG=0

#schedule = ["evo"+num,"base"+num,"EVO"+num]
#schedule = ["base"+num+"_"+str(q) for q in [0.0,0.25,0.5,0.75,1.0]]
AGENTS=5
ROBOTS=4
vals=sorted([0.8,1.0,0.6,0.3,0.2,0.1],reverse=True)
lbls={0:"D Rand.",1:"End State Aprx.",2:"Ctrfctl. Aprx.",3:"MTFC (Our Method)",4:"$D^\Sigma$",5:"$G^\Sigma$"}
if DEBUG:
    plt.subplot(1,2,1)
mint=1e9

for q in [3,4,5,1,2]:
    T=[]
    R=[]
    print(q)
    '''
    if (q !=3 and q!=1) or AGENTS==5:
        rmin,rmax=0,6
    else:
        rmin,rmax=12,18
    '''
    for i in range(12):
        log = logger.logger()
        
        try:
            log.load("tests/very/"+str(AGENTS)+"-"+str(ROBOTS)+"-"+str(i)+"-"+str(q)+".pkl")
        except:
            continue
    
        r=log.pull("reward")
        #L=log.pull("loss")
        t=log.pull("test")
        n_teams=len(t[0])
        print(n_teams)
        aprx=log.pull("aprx")
        if 0:
            for k in range(len(aprx)):
                print(k*1,k)
                arr=np.zeros((len(aprx[0]),AGENTS))
                for i in range(len(arr)):
                    team,vals=aprx[k][i]
                    vals=np.array(vals).T[0]
                    arr[i,team]=vals
                print(arr)    
                print(t[k])
        #print(t)
        r=np.array(r)

        t=np.array(t)
        mint=min(len(t),mint)
        
        print(np.round(t[-1,:],2))
        N=len(np.average(t,axis=0))
        t=np.sum(t,axis=1)
        if DEBUG:
            plt.plot(t)
        R.append(r)
        print(q,i,t[-1])
        T.append(t)
    if DEBUG:
        plt.subplot(1,2,2)

    
    #R=np.mean(R,axis=0)
    T=[t[:mint] for t in T]
    BEST=np.max(T,axis=0)
    std=np.std(T,axis=0)/np.sqrt(18)
    T=np.mean(T,axis=0)
    X=[i*1 for i in range(len(T))]
    #plt.subplot(2,1,1)
    #plt.plot(BEST)
    #plt.subplot(2,1,2)
    plt.plot(X,T,label=lbls[q])
    plt.fill_between(X,T-std,T+std,alpha=0.35, label='_nolegend_')

    #plt.ylim([0,1.15])
    plt.grid(True)
plt.legend([str(i) for i in range(8)])
max_val=sum(vals[:ROBOTS//2])*n_teams
#plt.plot(X,[0.5]*101,"--")
#plt.plot(X,[0.8]*101,"--")
#plt.legend(["Random Teaming + Types","Unique Learners","Types Only","Max single POI reward","Max reward"])
plt.xlabel("Generation")
plt.title("Performance of " + str(AGENTS)+" Agents, Teams of "+str(ROBOTS))
leg=plt.legend()
for legobj in leg.legendHandles:
    legobj.set_linewidth(4.0)
#leg=plt.legend(["Min","First Quartile","Median","Third Quartile","Max"])
#for legobj in leg.legendHandles:
#    legobj.set_linewidth(5.0)
plt.ylabel("$G^\Sigma$ of "+str(N)+" Teams")
print(len(T))
plt.plot([0,X[-1]],[max_val,max_val],"--",label="Max Score")
'''
if num[1]=="5":
    plt.title("5 agents, coupling req. of 2")
if num[1]=="8":
    plt.title("8 agents, coupling req. of 3")
'''
#plt.title("Team Performance Across \n Quartile Selection Methods")

plt.tight_layout()
#plt.savefig("figsv3/vis8_"+str(ROBOTS)+"_"+str(AGENTS)+".png")
plt.show()