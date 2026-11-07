"""Data integrity and maker toxicity split by the initiating order's lifespan."""
import numpy as np
from extract import load_tape,SOURCE,quote,dump,OUT
import json

def main():
    x=load_tape(SOURCE);o=x.orders;e=x.events;tr=x.trades
    fit=(o.t_us>=x.start_us+60e6)&(o.t_us<x.start_us+3590e6)
    g=o.loc[fit&~o.ioc]
    k=e.lifespan.to_numpy()[tr.taker_row.to_numpy()]
    y=quote(x,tr.t_us.to_numpy()+1000000,'mid')-tr.price.to_numpy()
    y*=np.where(tr.aggressor.to_numpy()=='A',1,-1)
    keep=(tr.t_us.to_numpy()>=x.start_us+60e6)&(tr.t_us.to_numpy()+1e6<x.start_us+3590e6)&np.isfinite(y)
    r={}
    for tag in ('G','F'):
        m=keep&(k==tag);q=tr.qty.to_numpy()[m]
        r[tag]=dict(matches=int(m.sum()),volume=int(q.sum()),maker_markout_1s=float(np.average(y[m],weights=q)) if len(q) else None)
    dump('data_quality_and_aggressors.json',dict(events=len(e),orders=len(o),matches=len(tr),
        sorted_timestamps=bool(e.t_us.is_monotonic_increasing),duplicate_timestamps=int(e.t_us.duplicated().sum()),
        duplicate_insert_order_ids=int(o.order_id.duplicated().sum()),nonpositive_prices=int((o.price<=0).sum()),
        nonpositive_sizes=int((o.volume<=0).sum()),fitting_limit_orders=len(g),fitting_crossing_limits=int(g.crossed.sum()),
        aggressor_type_markouts=r))
    print('DATA INTEGRITY COMPLETE',r,flush=True)
    # Equal-length count bins only. A partial final minute otherwise inflates Fano ratios.
    diag=json.loads((OUT/'train_diagnostics.json').read_text())
    ins=o.loc[fit]
    for seconds in (.01,.1,1,10,60):
        for tag,g in [('ioc',ins[ins.ioc]),('limit',ins[~ins.ioc])]:
            b=np.floor((g.t_us.to_numpy()-x.start_us-60e6)/(seconds*1e6)).astype(int)
            c=np.bincount(b,minlength=int(np.ceil(3530/seconds)))[:int(np.floor(3530/seconds))]
            diag[f'fano_{seconds:g}s'][tag]=float(c.var()/c.mean())
    diag['fano_note']='Only complete equal-exposure bins included.'
    dump('train_diagnostics.json',diag)

if __name__=='__main__':main()
