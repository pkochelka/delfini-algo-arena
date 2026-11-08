import sys,csv,numpy as np
D=sys.argv[1]
ev=[(float(l.split()[1]),int(l.split()[2])) for l in open(D+'.out') if l.startswith('VF')]
seg=[(float(r['start']),float(r['end'])) for r in csv.DictReader(open('marek/research/round2/s_dynamics/vol_segments.csv'))]
T=6000; g=np.arange(0,T,0.01); flag=np.zeros(g.size,bool); tru=np.zeros(g.size,bool)
cur=0;last=0
for t,v in ev:
    flag[int(last*100):int(t*100)]=cur; cur=v; last=t
flag[int(last*100):]=cur
for a,b in seg: tru[int(a*100):int(b*100)]=1
print('true vol frac %.3f flag frac %.3f  hit %.3f  false-on %.3f of calm'%(tru.mean(),flag.mean(),(flag&tru).sum()/tru.sum(),(flag&~tru).sum()/(~tru).sum()))
ons=[t for t,v in ev if v==1]; offs=[t for t,v in ev if v==0]
pnl=np.loadtxt(D+'/pnl.csv',delimiter=',',skiprows=1)
P=lambda t: np.interp(t,pnl[:,0],pnl[:,1])
for a,b in seg:
    on=[x for x in ons if x>=a-1]; on=on[0]-a if on else None
    off=[x for x in offs if x>=b-5]; off=off[0]-b if off else None
    print('%7.1f-%7.1f dur %5.1f onlag %6.2f offlag %6.2f  pnl[a-5,a] %6.0f [a,a+5] %6.0f [a+5,a+15] %6.0f  [b,b+10] %6.0f'%(a,b,b-a,on,off,P(a)-P(a-5),P(a+5)-P(a),P(a+15)-P(a+5),P(b+10)-P(b)))
