from .common import *
import numpy as np
from numba import njit
from experiments.growth_budget_lab.environment import initial, observation, step
from experiments.path_feedback_long_lab.kernel import draw
from experiments.behavioral_hormone_lab.runtime import tick as old_tick

@njit(cache=True)
def evaluate_old(task,seed,start,n,program,grids,states,lookup,starts):
    out=np.zeros((n,5));counts=np.zeros((n,9))
    for i in range(n):
        ep=start+i;s=initial(task,seed,ep,0,starts);mapid=-1
        if task==5:
            mapid=ep%len(starts);s[4]=starts[mapid,int(draw(seed,ep,0,1001)*4)]
        h=np.zeros((2,4))
        while not s[1]:
            obs=observation(task,seed,ep,0,s,grids,states)
            action,h,stats,receipt=old_tick(obs,program,h,4,draw(seed,ep,int(s[0]),1900+task))
            counts[i,action]+=1;counts[i,5]+=np.any(h!=0);counts[i,6]+=np.any(np.abs(h)>=1.)
            counts[i,7]+=receipt>=0;counts[i,8]+=stats[4]
            step(task,seed,ep,0,s,action,grids,states,lookup)
        out[i,0]=s[2];out[i,1]=s[3];out[i,2]=s[0];out[i,3]=mapid;out[i,4]=ep
    return out,counts,np.zeros(1)

@njit(cache=True)
def tick(obs,features,weights,emissions,h,mode,u,enabled=True,drop=-1,epsilon=0.):
    values=np.zeros(len(features))
    for i in range(len(features)):
        op,a,b,g=features[i]
        if g==drop and drop>=0:continue
        if op==0:v=obs[a]
        elif op==7:v=h[a] if mode else 0.
        elif op==1:v=values[a]*values[b]
        elif op==2:v=values[a]-values[b]
        elif op==3:v=values[a]+values[b]
        elif op==4:v=1. if values[a]>values[b] else 0.
        elif op==5:v=abs(values[a])
        else:v=min(1.,max(-1.,values[a]))
        values[i]=min(4.,max(-4.,v)) if np.isfinite(v) else 0.
    newh=np.zeros(4);rho=.9 if mode in (2,4) else 0.
    if mode:
        e=np.zeros(4)
        for k in range(4):
            if k//2!=drop:e[k]=min(1.,max(-1.,values[emissions[k]]))
        for k in range(4):
            if mode in (3,4):v=rho*.5*(h[k%2]+h[2+k%2])+e[k%2]+e[2+k%2]
            else:v=rho*h[k]+e[k]
            newh[k]=min(1.,max(-1.,v))
    # This is the declared graph router, and action identity belongs to a sink.
    # The environment never sees a score, feature vector or numeric readout.
    scores=np.zeros(len(weights))
    for a in range(len(weights)):
        for j in range(len(values)):scores[a]+=weights[a,j]*values[j]
    permit=-1
    if enabled:
        if epsilon>0 and u<epsilon:
            permit=min(len(scores)-1,int(u/epsilon*len(scores)))
        else:
            routed_u=(u-epsilon)/(1.-epsilon) if epsilon>0 else u
            high=scores.max();ties=np.empty(len(scores),np.int32);n=0
            for a in range(len(scores)):
                if scores[a]==high:ties[n]=a;n+=1
            permit=ties[min(n-1,int(routed_u*n))]
    action=0;receipt=-1
    # Only the uniquely authorized explicit zero-output sink can emit.
    for sink in range(len(weights)):
        if sink==permit:action=sink+1;receipt=sink+1
    return action,newh,receipt,values,scores

@njit(cache=True)
def evaluate(task,seed,start,n,features,weights,emissions,mode,enabled,grids,states,lookup,starts,drop=-1,random=False,epsilon=0.,stream=0,random_maps=False):
    seed=seed+stream*100000000
    out=np.zeros((n,5));counts=np.zeros((n,9));activity=np.zeros(len(features))
    for i in range(n):
        ep=start+i;s=initial(task,seed,ep,0,starts)
        mapid=-1
        if task==5:
            mapid=int(draw(seed,ep,0,1701)*len(starts)) if random_maps else ep%len(starts)
            d=int(draw(seed,ep,0,1001)*4);s[4]=starts[mapid,d]
        h=np.zeros(4)
        while not s[1]:
            obs=observation(task,seed,ep,0,s,grids,states)
            u=draw(seed,ep,int(s[0]),1900+task)
            action,h,receipt,v,scores=tick(obs,features,weights,emissions,h,mode,u,enabled,drop,epsilon)
            if random:
                # Random baseline is the same router with equal priorities,
                # implemented by calling tick with zero coefficients outside.
                assert receipt>0
            assert (receipt==-1 and action==0) or receipt==action
            counts[i,action]+=1;counts[i,5]+=np.any(h!=0);counts[i,6]+=np.any(np.abs(h)>=1.)
            counts[i,7]+=receipt>0;counts[i,8]+=np.count_nonzero(v)
            activity+=np.abs(v)>0
            step(task,seed,ep,0,s,action,grids,states,lookup)
        out[i,0]=s[2];out[i,1]=s[3];out[i,2]=s[0];out[i,3]=mapid;out[i,4]=ep
    return out,counts,activity

@njit(cache=True)
def trace_episode(task,seed,ep,features,weights,emissions,mode,enabled,grids,states,lookup,starts,epsilon=0.):
    s=initial(task,seed,ep,0,starts)
    if task==5:s[4]=starts[ep%len(starts),int(draw(seed,ep,0,1001)*4)]
    h=np.zeros(4);trace=np.zeros((1024,15));values=np.zeros((1024,len(features)))
    j=0
    while not s[1]:
        obs=observation(task,seed,ep,0,s,grids,states)
        action,h,receipt,v,scores=tick(obs,features,weights,emissions,h,mode,draw(seed,ep,int(s[0]),1900+task),enabled,-1,epsilon)
        values[j]=v;trace[j,0]=s[0];trace[j,1]=action;trace[j,2]=receipt;trace[j,3:7]=h
        trace[j,7:7+len(scores)]=scores
        step(task,seed,ep,0,s,action,grids,states,lookup)
        trace[j,11]=s[3];trace[j,12]=s[1];trace[j,13]=s[2];trace[j,14]=s[0];j+=1
    return trace[:j],values[:j]
