#for viewing teaming with pols
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.style.use('tableau-colorblind10')
#matplotlib.rcParams['text.usetex'] = True
from teaming import logger
'''
scp -J cookjos@access.engr.oregonstate.edu cookjos@graf200-17.engr.oregonstate.edu:robust_teaming/save/testing/* save/testing/
'''
for n_teams in [25,100]:
    for team_swap_frq in [500,5000]:
        
        T=[]
        mint=1e9
        for i in range(12):
            trial=i
            n_agents=4
            train_flag=4
            
        
        
    
            log = logger.logger()
            folder="save/testing/data"+"-".join([str(S) for S in [n_agents,n_teams,team_swap_frq,trial]])
            
            try:
                log.load(folder+".pkl")
                print("suc",folder+".pkl")
            except:
                
                print("err",folder+".pkl")
                continue

            t=log.pull("test")
            test_teams=len(t[0])
            #print(test_teams,i)

        

            t=np.array(t)
            mint=min(len(t),mint)
            
            
            #print(np.round(t[-1,:],2))
            print(np.sum(t[-1,:]))
            N=len(np.average(t,axis=0))
            t=np.sum(t,axis=1)
            T.append(t)


    
   
        T=[t[:mint] for t in T]
        BEST=np.max(T,axis=0)
        std=np.std(T,axis=0)/np.sqrt(4)
        T=np.mean(T,axis=0)
        X=[i*1 for i in range(len(T))]

        plt.plot(X,T,label="-".join([str(S) for S in [n_teams,team_swap_frq]]) )
        #plt.fill_between(X,T-std,T+std,alpha=0.35, label='_nolegend_')

        plt.grid(True)

plt.xlabel("Generation")
leg=plt.legend()
for legobj in leg.legendHandles:
    legobj.set_linewidth(4.0)

plt.ylabel("$G^\Sigma$ of "+str(N)+" Teams")


plt.tight_layout()
#plt.savefig("figsv3/vis8_"+str(ROBOTS)+"_"+str(AGENTS)+".png")
plt.show()