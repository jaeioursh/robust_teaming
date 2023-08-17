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


def load_data(n_agents=5,agent_idx=0,n_actors=4,iteration=0,generation=0,q=3,hidden=20):

    fname="tests/"+str(n_agents)+"-"+str(n_actors)+"-"+str(iteration)+"-"+str(q)

    log = logger.logger()
    log.load(fname+"/data.pkl")
    env=make_env(n_actors)
    pos=log.pull("position")
    teams=np.array(log.pull("types")[0])

    net=Net(hidden)
    net.model.load_state_dict(torch.load(fname+"/"+str(generation)+".mdl")[agent_idx])
    return env,pos,teams,net

#env.reset()
def eval(x,y,t,team_idx,agent_idx,env,position,teams,generation,time=-1):
    pos,rot=position[generation][team_idx][time]
    pos,rot=pos.copy(),rot.copy()
    pos[teams[team_idx]==agent_idx,:]=[x,y]
    env.data["Agent Orientations"]=rot
    env.data["Agent Orientations"][teams[team_idx]==agent_idx,:]=[np.sin(t),np.cos(t)]
    env.data["Agent Positions"]=pos
    env.data["Agent Position History"]=np.array([env.data["Agent Positions"]])
    env.data["Steps"]=0
    env.data["Observation Function"](env.data)
    z=env.data["Agent Observations"]
    s=z[teams[team_idx]==agent_idx]


    env.data["Reward Function"](env.data)
    g=env.data["Global Reward"]
    
    return s,g


if __name__=="__main__":
    agent_idx=0
    team_idx=0
    generation=0
    env,pos,teams,net=load_data(n_agents=5,agent_idx=agent_idx,n_actors=4,iteration=0,generation=generation)
    x=0 # -5 to 35 ish
    y=0 # -5 to 35 ish
    t=0 #-pi to pi
    state,G = eval(x,y,t,team_idx,agent_idx,env,pos,teams,generation,time=3)
    G_estimate=net.feed(state)[0,0]
    print(G,G_estimate)