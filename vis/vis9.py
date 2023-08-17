import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from teaming import logger
num="28"
schedule = ["evo"+num,"base"+num]
#schedule = ["evo"+num,"base"+num,"EVO"+num]
schedule = ["base"+num+"_"+str(q) for q in [0.0,0.25,0.5,0.75,1.0]]
schedule.append("evo28")
Y=[]
err=[]
for q in schedule:
    T=[]
    R=[]
    for i in range(8):
        log = logger.logger()
        
        log.load("tests/"+q+'-'+str(i)+".pkl")
       
        r=log.pull("reward")
        #L=log.pull("loss")
        t=log.pull("test")
        #print(t)
        r=np.array(r)

        t=np.array(t)

        if num[0]=="2":
            scale=0.64
        if num[0]=="1" or num[0]=="3":
            scale=0.8
        if num[0]=="0":
            vals=log.pull("poi vals")[0]
            print(vals)
            vals=sorted(vals,reverse=True)
            scale=(vals[0]+vals[1])

        N=len(np.average(t,axis=0))
        t=np.average(t,axis=1)/scale
        
        R.append(r)
        #T.append(max(t))
        T.append(t[-1])
    

    R=np.mean(R,axis=0)
    std=np.std(T)/np.sqrt(8)
    T=np.mean(T)
    Y.append(T)
    err.append(std)
    #plt.subplot(2,1,1)
    #plt.plot(R)
    #plt.subplot(2,1,2)
lbls=["0.0","0.25","0.5","0.75","1.0","mean"]
lbls=["Min","1st \nQuartile","Median","3rd \n Quartile","Max","Mean"]
plt.bar(lbls,Y,yerr=err)
plt.plot([4.5,4.5],[0,1])
plt.ylim([0.3,0.7])
plt.ylabel("Averaged Fitness Across "+str(N)+" Teams")
plt.xlabel("Aggregation Method")
plt.grid(axis="y")

#plt.title("Team Performance vs. Fitness Quartile Selection")
plt.tight_layout()
plt.savefig("figsv2/FFF_"+schedule[0][-2:]+".pdf")
plt.show()