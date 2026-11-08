import pickle,json,sys
for tape in sys.argv[1:]:
    X=pickle.load(open(f'b_{tape}.pkl','rb')); R=json.load(open(f'b_{tape}.pkl.rows.json'))
    for r in R:
        T=r['onset']+3
        tr=[x for x in X['trig'] if T-0.5<=x[0]<=T+5]
        if not tr: continue
        t,d,ref,sg,F=tr[0]
        if abs(r['Q'])<50 and r['V5']>1000:
            print(tape, round(r['onset'],2), 'J',round(r['J'],1),'V5',round(r['V5']),'tm',round(r['tm'],1),'cls',r['cls'],'pos0',r['p0'],'sg',None if sg is None else round(t-sg,2),'edge',round((F-ref)*d,1),'Q',r['Q'])
