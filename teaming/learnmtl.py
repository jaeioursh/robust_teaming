import re
import numpy as np
#import tensorflow as tf
import numpy as np

from copy import deepcopy as copy
from .logger import logger
import pyximport
from .cceamtl import *
from itertools import combinations
#from math import comb
from collections import deque
from random import sample
import torch
device = torch.device("cpu") 
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.set_num_threads(1)
print("threads: ",torch.get_num_threads())

import operator as op
from functools import reduce

def comb(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2


class Net():
    def __init__(self,hidden=20):#*4
        learning_rate=5e-3
        self.model = torch.nn.Sequential(
            torch.nn.Linear(8, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden,1)
        )
        self.loss_fn = torch.nn.MSELoss(reduction='sum')

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    def feed(self,x):
        x=torch.from_numpy(x.astype(np.float32))
        pred=self.model(x)
        return pred.detach().numpy()
        
    
    def train(self,x,y,n=5,verb=0):
        x=torch.from_numpy(x.astype(np.float32))
        y=torch.from_numpy(y.astype(np.float32))
        pred=self.model(x)
        loss=self.loss_fn(pred,y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()
    

def helper(t,k,n):
    if k==-1:
        return [t]
    lst=[]
    for i in range(n):
        if t[k+1]<=i:
            t[k]=i
            lst+=helper(copy(t),k-1,n)
    return lst



def robust_sample(data,n):
    if len(data)<n: 
        smpl=data
    else:
        smpl=sample(data,n)
    return smpl

class learner:
    def __init__(self,nagents,types,sim,external_agent,ext_params):
        self.log=logger()
        self.nagents=nagents
        
        self.external_agent=external_agent
        self.ext_params=ext_params
        self.itr=0
        self.types=types
        self.team=[]
        self.index=[]
        self.Dapprox=[Net() for i in range(self.nagents)]

        
        self.test_teams=self.set_test_teams()
        sim.data["Number of Policies"]=32

        self.hist=[deque(maxlen=self.types*2000) for i in range(self.nagents)]
        self.zero=[deque(maxlen=100) for i in range(self.nagents)]

        initCcea(input_shape=8, num_outputs=2, num_units=20, num_types=nagents)(sim.data)
        

    def act(self,S,data,trial):
        policyCol=data["Agent Policies"]
        policyCol[-1]=self.external_agent
        A=[]
        for i in range(len(policyCol)):
            s=S[i]
            pol=policyCol[i]
            if i==len(policyCol)-1:
                s[:4]=0
            a = pol.get_action(s)
            A.append(a)
        return np.array(A)*2.0
    

    def set_test_teams(self):
        team=[]

        for i in range(100):
            val=None
            while val is None:

                idx= tuple([np.random.randint(q) for q in self.ext_params.shape])
                val=self.ext_params[idx]
            team.append(idx)
        return team


            


    def set_teams(self,rand=0):
        self.team=[]

        for i in range(self.types):
            val=None
            while val is None:

                idx= tuple([np.random.randint(q) for q in self.ext_params.shape])
                val=self.ext_params[idx]
            self.team.append(idx)

        
        
        

    def set_single(self,team):
        params=self.ext_params[team][0]
        params=[np.copy(np.array(p)) for p in params]
        self.external_agent.__setstate__(params)


    def save(self,folder,net_save=True):
        print("saved")
        self.log.save(folder+".pkl")
        #print(self.Dapprox[0].model.state_dict()['4.bias'].is_cuda)
        if net_save:
            netinfo={i:self.Dapprox[i].model.state_dict() for i in range(len(self.Dapprox))}
            torch.save(netinfo,folder+".mdl")

    #train_flag=0 - D
    #train_flag=1 - Neural Net Approx of D
    #train_flag=2 - counterfactual-aprx
    #train_flag=3 - fitness critic
    #train_flag=4 - D*
    #train_flag=5 - G*
    def run(self,env,train_flag):

        populationSize=len(env.data['Agent Populations'][0])
        pop=env.data['Agent Populations']
        #team=self.team[0]
        G=[]
        
        for worldIndex in range(populationSize):
            env.data["World Index"]=worldIndex
            
            for team in self.team:
                self.set_single(team)
                s = env.reset() 
                done=False 
                #assignCceaPoliciesHOF(env.data)
                assignCceaPolicies(env.data)
                S,A=[],[]
                while not done:
                    self.itr+=1
                    
                    action=self.act(s,env.data,0)
                    S.append(s)
                    A.append(action)
                    s, r, done, info = env.step(action)
                #S,A=[S[-1]],[A[-1]]
                pols=env.data["Agent Policies"] 
                g=env.data["Global Reward"]
                for i in range(len(s)):

                    d=r[i]
                    
                    pols[i].G.append(g)
                    
                    pols[i].D.append(d)
                    pols[i].S.append([])
                    if train_flag==1 or train_flag==2 or train_flag==3:
                        for j in range(len(S)):
                            z=[S[j][i],A[j][i],g]
                            #if d!=0:
                            self.hist[i].append(z)
                            #else:
                            #    self.zero[team[i]].append(z)
                            pols[i].S[-1].append(S[j][i])
                        
                        pols[i].Z.append(S[-1][i])
                        
                G.append(g)
            

        if train_flag==1 or train_flag==2 or train_flag==3:
            self.updateD(env)
        
        for t in range(self.nagents):
            #if train_flag==1:
            #    S_sample=self.state_sample(t)

            for p in pop[t]:
                
                #d=p.D[-1]
                if train_flag==4:
                    p.fitness=np.sum(p.D)

                if  train_flag==5:
                    p.fitness=np.sum(p.G)

                if train_flag==3:
                    p.D=[self.Dapprox[t].feed(np.array(p.S[i])) for i in range(len(p.S))]
                    #self.log.store("ctime",[np.argmax(i) for i in p.D])
                    p.D=[np.max(i) for i in p.D]
                    #p.D=[(self.Dapprox[t].feed(np.array(p.S[i])))[-1] for i in range(len(p.S))]
                    #print(p.D)
                    p.fitness=np.sum(p.D)
                    
                if train_flag==1 or train_flag==2:
                    #self.approx(p,t,S_sample)
                    p.D=list(self.Dapprox[t].feed(np.array(p.Z)))
                    p.fitness=np.sum(p.D)
                    if train_flag==2:
                        p.fitness=np.sum(p.G)-np.sum(p.D)
                        
                    #print(p.fitness)

                    
                if train_flag==0:
                    d=p.D[-1]
                    p.fitness=d
                p.G=[]
                p.D=[]
                p.Z=[]
                p.S=[]
        evolveCceaPolicies(env.data)

        self.log.store("reward",max(G))      
        return max(G)


    def updateD(self,env):
        
        for i in range(self.nagents):
            for q in range(64):
                S,A,D=[],[],[]
                SAD=robust_sample(self.hist[i],100)
                #SAD+=robust_sample(self.zero[i],100)
                for samp in SAD:
                    S.append(samp[0])
                    A.append(samp[1])
                    D.append([samp[2]])
                S,A,D=np.array(S),np.array(A),np.array(D)
                Z=S#np.hstack((S,A))
                self.Dapprox[i].train(Z,D)
    def state_sample(self,t):
        S=[]
        A=[]
        SAD=robust_sample(self.hist[t],100)
        if len(SAD)==0:
            SAD+=robust_sample(self.zero[t],100)
        for samp in SAD:
            s=samp[0]
            S.append(s)
        return np.array(S)

    def approx(self,p,t,S):
        
        A=[p.get_action(s) for s in S]
        A=np.array(A)
        Z=np.hstack((S,A))
        D=self.Dapprox[t].feed(Z)
        fit=np.sum(D)
        #print(fit)
        p.fitness=fit

    def put(self,key,data):
        self.log.store(key,data)


    def test(self,env,itrs=50,render=0):

        old_team=self.team
        #
        

        self.log.clear("position")
        self.log.clear("types")
        
        self.log.clear("poi")
        self.log.store("poi",np.array(env.data["Poi Positions"]))
        self.log.clear("poi vals")
        self.log.store("poi vals",np.array(env.data['Poi Static Values']))
        Rs=[]
        teams=copy(self.test_teams)
        self.log.clear("teams")
        self.log.store("teams",teams)
        #self.log.store("idxs",self.index)

        aprx=[]
        team_pos=[]
        for i in range(len(teams)):

            
            
            #team=np.array(teams[i]).copy()ffmpeg -i input_file.mp4 -f mov output_file.mov
            #np.random.shuffle(team)
            self.team=[teams[i]]
            team=teams[i]
            self.set_single(team)
            #for i in range(itrs):
            assignBestCceaPolicies(env.data)
            #self.randomize()
            s=env.reset()
            done=False
            R=[]
            i=0
            
            positions=[[np.array(env.data["Agent Positions"]),np.array(env.data["Agent Orientations"])]]
            while not done:
                
                #self.log.store("position",np.array(env.data["Agent Positions"]),i)
                
                action=self.act(s,env.data,0)
                positions.append([np.array(env.data["Agent Positions"]),np.array(env.data["Agent Orientations"])])
                #action=self.idx2a(env,[1,1,3])
                #print(action)
                sp, r, done, info = env.step(action)
                if render:
                    env.render()
                
                s=sp
                i+=1
            team_pos.append(positions)
            g=env.data["Global Reward"]
            ap=[]
            #for t,State in zip(self.team[0],s):
            #    
            #    ap.append(self.Dapprox[t].feed(np.array(State)))
            #aprx.append([self.team[0],ap])
            Rs.append(g)
        self.log.store("position",team_pos)
        #self.log.store("aprx",aprx)
        self.log.store("test",Rs)
        self.aprx=aprx
        self.team=old_team

    

    def quick(self,env,episode,render=False):
        s=env.reset()
        
        for i in range(100):
            a=[[0,0] for i in range(self.nagents)]
            sp, r, done, info = env.step(a)
        return [0.0]
            
    
    
    


def test_net():
    a=Net()
    b=Net()
    x=np.array([[1,2,3,4,5,6,7,8]])
    y=np.array([[0]])
    print(a.feed(x))
    print(a.train(x,y))
    print(b.feed(x))
    print(b.train(x,y))

if __name__=="__main__":
    test_net()
    a=all_teams(5)
    print(a)
    
    