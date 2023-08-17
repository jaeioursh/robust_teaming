"""
An example using the rover domain gym-style interface and the standard, included CCEA learning algorithms.
This is a minimal example, showing the minimal Gym interface.
"""
from os import killpg
import os
import numpy as np
import sys
import multiprocessing as mp


from rover_domain_core_gym import RoverDomainGym
import pyximport
pyximport.install()
from teaming.cceamtl import *
import code.agent_domain_2 as domain

#import mods
from teaming.learnmtl import learner
from sys import argv
import pickle
#import tensorflow as tf

def rand_loc(n):
    x,y=np.random.random(2)
    pos=[[x,y]]
    while len(pos)<6:
        X,Y=np.random.random(2)
        for x,y in pos:
            dist=((X-x)**2.0+(Y-y)**2.0 )**0.5
            if dist<0.2:
                X=None 
                break
        if not X is None: 
            pos.append([X,Y])
    
    return np.array(pos)


#print(vals)
def make_env(nagents,coup=2,rand=0):
    vals =np.array([0.8,1.0,0.6,0.3,0.2,0.1])
    
    if rand:
        pos=np.array([
            [0.0, 0.2],
            [0.7, 0.1],
            [1.0, 0.3],
            [0.4, 0.6],
            [0.3, 0.3],
            [0.4, 0.9]
        ])
    else:
        pos=np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.5],
            [0.0, 0.5],
            [1.0, 0.0]
        ])
    
    #pos=rand_loc(6)#np.random.random((6,2))
    #vals=np.random.random(6)/2.0
    print(vals)

    sim = RoverDomainGym(nagents,30,pos,vals)
 


    sim.data["Coupling"]=coup
    sim.data['Number of Agents']=nagents

    obs=sim.reset()

    sim.data['Agent Positions BluePrint'][-1]=[15.0,15.0]
    ang=np.pi/2
    sim.data['Agent Orientations BluePrint'][-1]=[np.cos(ang),np.sin(ang)]
    obs=sim.reset()
    return sim


import time

def test1(trial,k,n,train_flag,n_teams):
    #print(np.random.get_state())[1]    
    np.random.seed(int(time.time()*100000)%100000)
    env=make_env(n)

    with open("save/a.pkl","rb") as f:
        arry = pickle.load(f)
    agent=Evo_MLP(8,2,20)

 
    OBS=env.reset()

    controller = learner(n,k,env,agent,arry)
    #controller.set_teams(n_teams)

    for i in range(4001):

        
        #controller.randomize()
        if i%100000==0:
            controller.set_teams(n_teams)

        if i%1==0:
            controller.test(env)

        r=controller.run(env,train_flag)# i%100 == -10)
        print(i,r,len(controller.team),train_flag)
        
            
        if i%50==0:
           
            
            folder="tests/"+str(k)+"-"+str(n)+"-"+str(trial)+"-"+str(train_flag)
            if not os.path.exists("tests"):
                os.makedirs("tests")
            if not os.path.exists(folder):
                os.makedirs(folder)
            controller.save(folder,i)

    #train_flag=0 - D
    #train_flag=1 - Neural Net Approx of D
    #train_flag=2 - counterfactual-aprx
    #train_flag=3 - fitness critic
    #train_flag=4 - D*
    #train_flag=5 - G*
if __name__=="__main__":
    if 0:
        import cProfile, pstats, io
        from pstats import SortKey
        pr = cProfile.Profile()
        pr.enable()
        # ... do something ...
        test1(42,5,4,1)
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        
    else:
        for train in [4]:
            procs=[]
            k=5
            n=4
            for k,n in [[5,4]]:
                teams=100
                for i in range(1):
                    
                    p=mp.Process(target=test1,args=(i,k,n,train,teams))
                    p.start()
                    time.sleep(0.05)
                    procs.append(p)
                    #p.join()git ad
                for p in procs:
                    p.join()

# 100 - static
# 200 - minimax single
# 300 random
# 400 most similar