import csv,sys
R='marek/research/round2/flow/'
shocks=[float(r['onset']) for r in csv.DictReader(open(R+'shock_events.csv'))]
segs=[(float(r['start']),float(r['end']),r['high']=='True') for r in csv.DictReader(open(R+'taker_regimes.csv'))]
def lab(t):
    for s in shocks:
        if s<=t<s+20: return 'shock(0..20s)'
    for a,b,h in segs:
        if a<=t<b: return 'volatile' if h else 'calm'
    return 'calm'
for name in ['r2_s_ladder','r2_sig_ladder','r2_robust_ladder']:
    rows=list(csv.DictReader(open(f'marek/logs/{name}/pnl.csv')))
    acc={};secs={}
    prev=0.0
    for r in rows[1:]:
        t=float(r['elapsed_s']);p=float(r['pnl']);d=p-prev;prev=p
        l=lab(t-0.5);acc[l]=acc.get(l,0)+d;secs[l]=secs.get(l,0)+1
    print(name, {k:(round(v),secs[k]) for k,v in acc.items()}, 'total',round(prev))
print('--- shock sub-windows (s_ladder), and volatile by time since entry')
rows=list(csv.DictReader(open('marek/logs/r2_s_ladder/pnl.csv')))
pn={round(float(r['elapsed_s'])):float(r['pnl']) for r in rows}
b={}
for s in shocks:
    t0=int(s)
    for a,bb,n in [(-1,3,'drought'),(3,5,'jump'),(5,10,'burst'),(10,20,'after')]:
        b[n]=b.get(n,0)+pn.get(t0+bb,0)-pn.get(t0+a,0)
print({k:round(v) for k,v in b.items()})
v={}
for a,bb,h in segs:
    if not h: continue
    a=int(a);bb=int(bb)
    for lo,hi,n in [(0,2,'first2s'),(2,10,'2-10s'),(10,10**6,'>10s')]:
        x=min(a+hi,bb);y=a+lo
        if x>y: v[n]=v.get(n,0)+pn.get(x,0)-pn.get(y,0)
print({k:round(val) for k,val in v.items()})
