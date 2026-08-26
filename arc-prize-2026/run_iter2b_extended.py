"""Iter 2b: Extended test on the 10 games that completed L1, with 10 min each.
Goal: See if more time = more levels, and measure actions/sec on GPU."""

import json, time, datetime, hashlib, random
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import arc_agi
from arcengine.enums import GameAction, GameState

ACTION_MAP = {a.value: a for a in GameAction}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)

# Same ActionModel as before
class ActionModel(nn.Module):
    def __init__(self, nc=16, gs=64):
        super().__init__()
        self.conv1=nn.Conv2d(nc,32,3,padding=1); self.conv2=nn.Conv2d(32,64,3,padding=1)
        self.conv3=nn.Conv2d(64,128,3,padding=1); self.conv4=nn.Conv2d(128,256,3,padding=1)
        self.ap=nn.MaxPool2d(4,4); self.afc=nn.Linear(256*16*16,512)
        self.ah=nn.Linear(512,5); self.do=nn.Dropout(0.2)
        self.cc1=nn.Conv2d(256,128,3,padding=1); self.cc2=nn.Conv2d(128,64,3,padding=1)
        self.cc3=nn.Conv2d(64,32,1); self.cc4=nn.Conv2d(32,1,1)
    def forward(self,x):
        x=F.relu(self.conv1(x));x=F.relu(self.conv2(x));x=F.relu(self.conv3(x))
        f=F.relu(self.conv4(x));a=self.ap(f).view(f.size(0),-1)
        a=self.do(F.relu(self.afc(a)));al=self.ah(a)
        c=F.relu(self.cc1(f));c=F.relu(self.cc2(c));c=F.relu(self.cc3(c));c=self.cc4(c)
        return torch.cat([al,c.view(c.size(0),-1)],dim=1)

def onehot(grid,dev):
    f=np.array(grid,dtype=np.int64);
    if f.ndim==3:f=f[-1]
    t=torch.zeros(16,64,64,dtype=torch.float32)
    t.scatter_(0,torch.from_numpy(f).unsqueeze(0).clamp(0,15),1)
    return t.to(dev)

def sample(model,ft,avail,dev):
    NC=4096
    with torch.no_grad(): lg=model(ft.unsqueeze(0)).squeeze(0)
    al,cl=lg[:5],lg[5:]
    m=torch.full((5,),float('-inf'),device=dev);a6=False
    for a in avail:
        v=a.value if hasattr(a,'value') else a
        if 1<=v<=5:m[v-1]=0.
        elif v==6:a6=True
    al=al+m
    if not a6:cl=cl+torch.full_like(cl,float('-inf'))
    ap=torch.sigmoid(al);cp=torch.sigmoid(cl)/NC
    p=torch.cat([ap,cp]);s=p.sum()
    if s<1e-10:p=torch.ones_like(p);p[5:]=1./NC;s=p.sum()
    p=p/s;return np.random.choice(len(p.cpu().numpy()),p=p.cpu().numpy())

# Games that completed L1 in iter2 (sorted by speed)
PROMISING = ["r11l","lp85","sp80","m0r0","ft09","cn04","cd82","ar25","sk48","vc33"]
TIME_PER_GAME = 600  # 10 minutes each

print(f"Device: {DEVICE}")
print(f"Extended test: {len(PROMISING)} games, {TIME_PER_GAME}s each")
print(f"Estimated total: {len(PROMISING)*TIME_PER_GAME/60:.0f} min\n")

arcade = arc_agi.Arcade()
envs = {e.game_id.split('-')[0]: e for e in arcade.get_environments()}
results = []

for gi, gname in enumerate(PROMISING):
    ei = [e for k,e in envs.items() if k.startswith(gname)]
    if not ei: print(f"Skip {gname}"); continue
    env_info = ei[0]
    print(f"\n[{gi+1}/{len(PROMISING)}] {env_info.title} ({','.join(env_info.tags)})")
    print(f"  Levels: {len(env_info.baseline_actions)}, Human: {sum(env_info.baseline_actions)}")

    env=arcade.make(env_info.game_id); frame=env.reset()
    avail=[ACTION_MAP[a] for a in frame.available_actions]
    model=ActionModel().to(DEVICE); opt=optim.Adam(model.parameters(),lr=1e-4)
    buf=deque(maxlen=200000); hashes=set()
    cf=onehot(frame._frame[0],DEVICE); pnp=cf.cpu().numpy().astype(bool)
    pidx=None; clvl=0; acts=0; t0=time.time(); lvl_times=[]

    while time.time()-t0<TIME_PER_GAME:
        if frame.levels_completed>clvl:
            el=time.time()-t0
            lvl_times.append({"level":frame.levels_completed,"action":acts,"time":round(el,1)})
            print(f"  *** Level {frame.levels_completed} at action {acts} ({el:.0f}s)")
            clvl=frame.levels_completed; buf.clear(); hashes.clear()
            model=ActionModel().to(DEVICE); opt=optim.Adam(model.parameters(),lr=1e-4)
            pnp=None; pidx=None
        if frame.state in (GameState.NOT_PLAYED,GameState.GAME_OVER):
            frame=env.step(GameAction.RESET);pnp=None;pidx=None;continue
        if frame.state==GameState.WIN:
            print(f"  WIN!");break
        cf=onehot(frame._frame[0],DEVICE);cnp=cf.cpu().numpy().astype(bool)
        if pnp is not None and pidx is not None:
            h=hashlib.md5(pnp.tobytes()+str(pidx).encode()).hexdigest()
            if h not in hashes:
                ch=not np.array_equal(pnp,cnp)
                buf.append({'state':pnp,'action_idx':pidx,'reward':1. if ch else 0.})
                hashes.add(h)
        idx=sample(model,cf,avail,DEVICE)
        if idx<5:frame=env.step(ACTION_MAP[idx+1])
        else:
            ci=idx-5;y,x=ci//64,ci%64
            frame=env.step(ACTION_MAP[6],data={'x':int(x),'y':int(y)})
        pnp=cnp;pidx=idx;acts+=1
        if acts%5==0 and len(buf)>=64:
            model.train();bi=np.random.choice(len(buf),64,replace=False)
            b=[buf[i] for i in bi]
            s=torch.stack([torch.from_numpy(e['state']).float().to(DEVICE) for e in b])
            ai=torch.tensor([e['action_idx'] for e in b],dtype=torch.long,device=DEVICE)
            r=torch.tensor([e['reward'] for e in b],dtype=torch.float32,device=DEVICE)
            opt.zero_grad();lg=model(s);sl=lg.gather(1,ai.unsqueeze(1)).squeeze(1)
            lo=F.binary_cross_entropy_with_logits(sl,r)
            ap=torch.sigmoid(lg);lo=lo-1e-4*ap[:,:5].mean()-1e-5*ap[:,5:].mean()
            lo.backward();opt.step();model.eval()
        if acts%2000==0:
            el=time.time()-t0
            print(f"    {acts} actions, {len(buf)} exp, lvls={frame.levels_completed}, "
                  f"{acts/el:.0f} act/s, {el:.0f}s")

    el=time.time()-t0
    r={"title":env_info.title,"levels":frame.levels_completed,
       "win_levels":len(env_info.baseline_actions),"actions":acts,
       "elapsed":round(el,1),"acts_per_sec":round(acts/el,1),
       "level_times":lvl_times,"human_baseline":sum(env_info.baseline_actions)}
    results.append(r)
    print(f"  Result: {r['levels']}/{r['win_levels']} levels, {acts} actions, "
          f"{r['acts_per_sec']} act/s")

# Summary
print("\n"+"="*70)
print("EXTENDED TEST SUMMARY")
print("="*70)
tl=sum(r['levels'] for r in results)
print(f"Total levels: {tl}")
for r in results:
    lvls=f"  L{[l['level'] for l in r['level_times']]}" if r['level_times'] else ""
    print(f"  {r['title']:5s}: {r['levels']}/{r['win_levels']} lvls, "
          f"{r['actions']} acts ({r['acts_per_sec']} act/s){lvls}")

with open("data/iter2b_extended.json","w") as f:
    json.dump({"results":results,"time_per_game":TIME_PER_GAME,"device":DEVICE,
               "total_levels":tl},f,indent=2)
print(f"\nSaved to data/iter2b_extended.json")
