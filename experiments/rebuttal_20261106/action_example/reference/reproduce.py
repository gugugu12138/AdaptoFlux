"""Portable independent interpreter: Python + NumPy, no ATF or Numba imports.

Run from the supplied model bundle:
python reproduce.py --model models/system_s104.json --world world_test.npz --episodes 2048
The actual representative system is recorded in frozen_manifest.json.
"""
import argparse,json,math
from pathlib import Path
import numpy as np

DX=(0,1,0,-1);DY=(-1,0,1,0)

def draw(seed,ep,index,stream):
    x=((seed+1)*1000003+(ep+1)*91771+(index+1)*99991+stream*67867967)&0xffffffff
    x=((x^(x>>16))*0x7feb352d)&0xffffffff
    x=((x^(x>>15))*0x846ca68b)&0xffffffff
    x=x^(x>>16)
    return (x+.5)/4294967296.

def food(s,seed,ep,number):
    length=int(s[6]);ix=int(draw(seed,ep,number,300)*(64-length))
    occupied={(int(s[11+k]),int(s[75+k])) for k in range(length)}
    for y in range(8):
        for x in range(8):
            if (x,y) not in occupied:
                if ix==0:return x,y
                ix-=1
    return -1,-1

def initialize(task,seed,ep,w):
    s=np.zeros(140)
    if task==0:
        s[4]=3;sc=int(draw(seed,ep,0,200)*14);s[100]=sc//2;s[101]=(sc%2)*2-1
    elif task==5:s[4]=w['starts'][ep%len(w['starts']),int(draw(seed,ep,0,1001)*4)]
    elif task==6:
        s[6]=3;s[7]=1;s[11:14]=[4,3,2];s[75:78]=4;s[9],s[10]=food(s,seed,ep,0)
    else:s[4]=.5
    return s

def observe(task,seed,ep,s,w):
    o=np.zeros(157);o[4]=1;o[156]=s[0]
    if task==0:
        o[0]=(s[100]-3)/3;o[1]=(s[4]-3)/3;o[2]=s[101];o[3]=(10-int(s[0])%10)/10
    elif task==5:
        m,x,y,d=map(int,w['states'][int(s[4])]);o[0]=.75;o[1]=(x-4)/4;o[2]=.75;o[3]=(y-4)/4
        for j in range(3):dd=(d+j-1)%4;o[5+j]=1-w['grids'][m,y+DY[dd],x+DX[dd]]
        o[8]=DX[d];o[9]=DY[d];o[74:155]=w['grids'][m].ravel()
    elif task==6:
        x,y,d,length=int(s[11]),int(s[75]),int(s[7]),int(s[6])
        o[0]=(s[9]-3.5)/3.5;o[1]=(x-3.5)/3.5;o[2]=(s[10]-3.5)/3.5;o[3]=(y-3.5)/3.5
        occupied={(int(s[11+k]),int(s[75+k])) for k in range(length-1)}
        for j in range(3):
            dd=(d+j-1)%4;nx=x+DX[dd];ny=y+DY[dd]
            o[5+j]=float(nx<0 or nx>=8 or ny<0 or ny>=8 or (nx,ny) in occupied)
        o[8]=DX[d];o[9]=DY[d]
        for j in range(length):o[10+int(s[75+j])*8+int(s[11+j])]=1
        o[155]=s[6]
    else:
        center=.25+.5*draw(seed,ep,int(s[0])//32,400)
        o[0]=s[4];o[1]=center;o[2]=s[5]*10;o[3]=(32-int(s[0])%32)/32
        o[5]=center-.18;o[6]=center+.18;o[7]=s[4]-center
    return o

def advance(task,seed,ep,s,action,w):
    if task==0:
        if action==1:s[4]=max(0,s[4]-1)
        elif action==2:s[4]=min(6,s[4]+1)
        if not 0<=s[100]+s[101]<=6:s[101]=-s[101]
        s[100]+=s[101];s[0]+=1
        if int(s[0])%10==0:
            s[8]+=abs(s[4]-s[100])<=1;s[3]=s[8]/16
            sc=int(draw(seed,ep,int(s[0])//10,200)*14);s[100]=sc//2;s[101]=(sc%2)*2-1
        if s[0]>=160:s[1]=1;s[2]=float(s[8]==16)
    elif task==5:
        m,x,y,d=map(int,w['states'][int(s[4])])
        if action==1:d=(d-1)%4
        elif action==2:d=(d+1)%4
        elif action==3:
            nx=x+DX[d];ny=y+DY[d]
            if w['grids'][m,ny,nx]:x,y=nx,ny
        s[4]=w['lookup'][m,y,x,d];s[0]+=1
        if x==7 and y==7:s[1]=s[2]=s[3]=1
        elif s[0]>=1024:s[1]=1
    elif task==6:
        length=int(s[6]);d=int(s[7])
        if action==1:d=(d-1)%4
        elif action==2:d=(d+1)%4
        s[7]=d;nx=int(s[11])+DX[d];ny=int(s[75])+DY[d];eating=nx==s[9] and ny==s[10]
        collision=not(0<=nx<8 and 0<=ny<8)
        for k in range(length if eating else length-1):collision=collision or (s[11+k]==nx and s[75+k]==ny)
        s[0]+=1
        if collision:s[1]=1;return
        if eating:length+=1
        for k in range(length-1,0,-1):s[11+k]=s[11+k-1];s[75+k]=s[75+k-1]
        s[11]=nx;s[75]=ny;s[6]=length
        if eating:
            s[8]+=1;s[3]=s[8]/8
            if s[8]>=8:s[1]=s[2]=1;return
            s[9],s[10]=food(s,seed,ep,int(s[8]))
        if s[0]>=512:s[1]=1
    else:
        tick=int(s[0]);center=.25+.5*draw(seed,ep,tick//32,400)
        s[5]=-.035 if action==1 else min(.08,s[5]+.003);s[4]+=s[5];s[0]+=1
        if s[4]<.02 or s[4]>.98:s[1]=1;return
        if tick%32==31:
            if abs(s[4]-center)>.16:s[1]=1;return
            s[8]+=1;s[3]=s[8]/32
        if s[0]>=1024:s[1]=1;s[2]=float(s[8]==32)

def graph_tick(obs,rec,h,u):
    values=[]
    for op,a,b,owner in rec['features']:
        if op==0:v=float(obs[a])
        elif op==7:v=h[a] if rec['mode'] else 0.
        elif op==1:v=values[a]*values[b]
        elif op==2:v=values[a]-values[b]
        elif op==3:v=values[a]+values[b]
        elif op==4:v=float(values[a]>values[b])
        elif op==5:v=abs(values[a])
        else:v=min(1.,max(-1.,values[a]))
        values.append(min(4.,max(-4.,v)) if math.isfinite(v) else 0.)
    emissions=[min(1.,max(-1.,values[i])) for i in rec['emissions']]
    newh=[0.]*4;mode=rec['mode'];rho=.9 if mode in (2,4) else 0.
    for k in range(4):
        if mode in (3,4):v=rho*.5*(h[k%2]+h[2+k%2])+emissions[k%2]+emissions[2+k%2]
        elif mode:v=rho*h[k]+emissions[k]
        else:v=0.
        newh[k]=min(1.,max(-1.,v))
    priorities=[sum(w*v for w,v in zip(row,values)) for row in rec['weights']]
    permit=-1
    if rec['enabled']:
        epsilon=rec.get('epsilon',0.)
        if epsilon and u<epsilon:permit=min(len(priorities)-1,int(u/epsilon*len(priorities)))
        else:
            u=(u-epsilon)/(1-epsilon) if epsilon else u
            ties=[i for i,p in enumerate(priorities) if p==max(priorities)]
            permit=ties[min(len(ties)-1,int(u*len(ties)))]
    emitted=[]
    for sink in range(len(rec['action_nodes'])):
        if sink==permit:emitted.append(sink+1)
    assert len(emitted)<=1
    return emitted[0] if emitted else 0,newh

def episode(rec,seed,ep,w,collect=False):
    task=rec['task'];s=initialize(task,seed,ep,w);h=[0.]*4;trace=[]
    while not s[1]:
        obs=observe(task,seed,ep,s,w);action,h=graph_tick(obs,rec,h,draw(seed,ep,int(s[0]),1900+task))
        advance(task,seed,ep,s,action,w)
        if collect:trace.append([action,*h,s[3],s[1],s[2],s[0]])
    return [s[2],s[3],s[0]],np.asarray(trace)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--world',required=True)
    p.add_argument('--episodes',type=int,default=32);p.add_argument('--start',type=int,default=83000000);p.add_argument('--output')
    a=p.parse_args();system=json.loads(Path(a.model).read_text(encoding='utf-8'));w=dict(np.load(a.world));result={}
    for name,entry in system['modules'].items():
        out=np.array([episode(entry['model'],system['seed']+system.get('environment_seed_offset',0),a.start+i,w)[0] for i in range(a.episodes)])
        result[name]={'complete_success':float(out[:,0].mean()),'progress':float(out[:,1].mean()),'episodes':a.episodes}
        print(name,result[name],flush=True)
    result['S']=sum(x['complete_success'] for x in result.values())/4
    print('S =',result['S'],'target met =',result['S']>=.8)
    if a.output:
        dest=Path(a.output)
        if dest.exists():raise FileExistsError('Choose a fresh output path')
        dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
