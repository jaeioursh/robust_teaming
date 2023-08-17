import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from teaming import logger
PLOT=0
DATA=[[] for i in range(3)]
n_agents=[4,8,16]
n_types=[2,3,4,5]
for k in n_agents:
    for n in n_types:
        PLOT+=1
        T=[]
        R=[]
        for i in range(8):
            log = logger.logger()
            
            log.load("tests/vary/"+str(k)+"-"+str(n)+"-"+str(i)+".pkl")
            
            r=log.pull("reward")
            #L=log.pull("loss")
            t=log.pull("test")
            #print(t)
            r=np.array(r)

            t=np.array(t)

            if k==16:
                scale=0.64*0.8
            if k==8:
                scale=0.64
            if k==4:
                scale=0.64

            N=len(np.average(t,axis=0))
            t=np.average(t,axis=1)/scale
            
            R.append(np.max(r))
            T.append(t)
        convert={4:0,8:1,16:2}
        idx=convert[k]
        print(R)
        R=np.mean(R,axis=0)
        T=np.max(T,axis=0)
        T=np.mean(T,axis=0)
        #T=T[-1]
        DATA[idx].append(T)

        if 0:
            std=np.std(T,axis=0)/np.sqrt(8)
            T=np.mean(T,axis=0)
            X=[i*50 for i in range(len(T))]
            #plt.subplot(2,1,1)
            #plt.plot(R)
            #plt.subplot(2,1,2)
            plt.subplot(3,4,PLOT)
            plt.plot(X,T)
            plt.fill_between(X,T-std,T+std,alpha=0.35)

            plt.ylim([0,1.1])
            plt.grid(True)
            #plt.plot(X,[0.5]*101,"--")
            #plt.plot(X,[0.8]*101,"--")
            #plt.legend(["Random Teaming + Types","Unique Learners","Types Only","Max single POI reward","Max reward"])
            plt.xlabel("Generation")
            #leg=plt.legend(["Min","First Quartile","Median","Third Quartile","Max"])
            #for legobj in leg.legendHandles:
            #    legobj.set_linewidth(5.0)
            plt.ylabel("Average Score Across "+str(N)+" Teams")

print(DATA)
m=["o","^","D"]
for i in range(3):
    d=DATA[i]
    x= np.arange(len(d))  # the label locations
    width = 0.25
    #plt.bar(x + (i-1)*width,d, width, label=str(n_agents[i])+" agents")
    plt.plot(x,d,marker=m[i], label=str(n_agents[i])+" robots")
labels=[str(i) for i in n_types]
ax=plt.gca()
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.ylim(0.2,0.8)
plt.xlabel("Number of Agents")
plt.ylabel("Performance")
#plt.title("Team Performance Across \n Quartile Selection Methods")

#plt.tight_layout()
plt.grid(axis="y")
plt.savefig("figsv2/vary.pdf")
plt.show()