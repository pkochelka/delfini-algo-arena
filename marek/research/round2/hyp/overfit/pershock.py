# per-shock PnL attribution: window [onset-0.5, onset+6] s per truth shock (synth) / r2 shock list, for each overfit run
import json,glob,os,sys,bisect
T='/var/folders/8l/1449kxk54p57ztrxc2br5b300000gn/T'
def find():
    m={}
    for d in glob.glob(T+'/hyp_*'):
        p=d+'/overfit.py'
        if not os.path.exists(p): continue
        src=open(p).read(); i=src.rfind("\n\nPARAMS.update({")
        key=src[i:].strip() if src.rstrip().endswith('})') and i>0 and 'xs_' not in '' else ''
        tail=src.splitlines()[-1]
        m['BASE' if 'kick' in tail else tail]=d
    return m
def series(d,t):
    rows=[l.split(',') for l in open(f'{d}/{t}/pnl.csv').read().split('\n')[1:] if l]
    return [float(a) for a,b in rows],[float(b) for a,b in rows]
def at(ts,ps,x): return ps[max(0,bisect.bisect_right(ts,x)-1)]
m=find(); print(list(m))
b=m['BASE']
for k,d in m.items():
    if k=='BASE' or not os.path.exists(d+'/s7/pnl.csv'): continue
    tot={'big':0,'small':0,'other':0}; nb=ns=0
    for s in range(1,8):
        tr=json.load(open(f'data/synth/r2_v2_seed{s}.truth.json'))
        tb,pb=series(b,f's{s}'); tx,px=series(d,f's{s}')
        sh=0
        for o,j in zip(tr['onsets'],tr['jumps']):
            dd=(at(tx,px,o+6)-at(tx,px,o-0.5))-(at(tb,pb,o+6)-at(tb,pb,o-0.5)); sh+=dd
            if abs(j)>=60: tot['big']+=dd; nb+=1
            else: tot['small']+=dd; ns+=1
        tot['other']+=(px[-1]-pb[-1])-sh
    print(k, {a:round(v) for a,v in tot.items()}, 'n_big',nb,'n_small',ns)
# r2: other 26 shocks vs 5398
on=[]
for l in open('marek/research/round2/hyp/shocktime/join_base.txt').read().splitlines()[1:]:
    try: on.append((float(l.split()[0]),float(l.split()[1])))
    except: pass
print('r2 shocks', len(on))
for k,d in m.items():
    if k=='BASE' or not os.path.exists(d+'/r2/pnl.csv'): continue
    tb,pb=series(b,'r2'); tx,px=series(d,'r2'); r={'5398':0,'other_shocks':0}
    for o,j in on:
        dd=(at(tx,px,o+6)-at(tx,px,o-0.5))-(at(tb,pb,o+6)-at(tb,pb,o-0.5))
        r['5398' if abs(o-5395.8)<2 else 'other_shocks']+=dd
    r['nonshock']=(px[-1]-pb[-1])-r['5398']-r['other_shocks']
    print('r2',k,{a:round(v) for a,v in r.items()})
