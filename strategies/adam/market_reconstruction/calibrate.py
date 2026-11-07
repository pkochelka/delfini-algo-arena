"""Extract state-conditioned marks from training data; keep validation periods separate."""
import csv
import json
import sys
from array import array
from pathlib import Path

import numpy as np

from engine import Book, Recorder, summarize

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').exists() and (p / 'simulator').is_dir())
OUT = HERE/'results'
OUT.mkdir(exist_ok=True)
TRAIN_END, DEV_END, END = 7200.,9600.,12000.


def main():
    source = ROOT/'data'/'round1.csv'
    book,rec = Book(),Recorder(END)
    g = {key:array('d') for key in ('t','side','size','offset','spread','share','life')}
    ioc = {key:array('d') for key in ('t','side','size','offset','spread','share')}
    id_to_g = array('q')
    first_id = None
    with source.open(newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for row_number,row in enumerate(reader):
            tm,op,oid,ss,vv,pp,life = row
            t,oid = float(tm),int(oid)
            rec.advance(t,book)
            if op=='Cancel':
                rec.cancel(t,oid,book)
                j = oid-first_id
                if 0<=j<len(id_to_g) and id_to_g[j]>=0:
                    index = id_to_g[j]
                    g['life'][index] = t-g['t'][index]
                continue
            if first_id is None:
                first_id = oid
            assert oid==first_id+len(id_to_g), 'Source IDs are not consecutive; use a general mapping'
            side,vol,price = (0 if ss=='B' else 1),int(vv),int(pp)
            bid,ask = book.best(0),book.best(1)
            own,opp = book.best(side),book.best(1-side)
            spread = ask-bid if bid and ask else np.nan
            share = book.qty(side)/(book.qty(0)+book.qty(1)) if bid and ask else np.nan
            if life=='G':
                index = len(g['t'])
                offset = (own-price)*(1 if side==0 else -1) if own else np.nan
                values = (t,side,vol,offset,spread,share,np.nan)
                for key,value in zip(g,values):
                    g[key].append(value)
                id_to_g.append(index)
            else:
                offset = (price-opp)*(1 if side==0 else -1) if opp else np.nan
                for key,value in zip(ioc,(t,side,vol,offset,spread,share)):
                    ioc[key].append(value)
                id_to_g.append(-1)
            result = rec.insert(t,oid,side,price,vol,life=='F',book)
            assert life=='F' or result[0]==0, 'Unexpected aggressive G insert'
            if row_number and row_number%500_000==0:
                print(f'Replayed {row_number:,} source rows at {t:.1f}s',flush=True)
    rec.advance(END,book)
    rec.save(OUT/'original_measurements.npz')
    g = {k:np.asarray(v) for k,v in g.items()}
    ioc = {k:np.asarray(v) for k,v in ioc.items()}
    # Exclude boundary-censored cancellation marks and the first 30 seconds of startup.
    keep = (g['t']>=30)&(g['t']<TRAIN_END-10)&np.isfinite(g['offset'])&np.isfinite(g['spread'])&np.isfinite(g['life'])
    keep &= g['t']+g['life']<TRAIN_END
    gfit = {key:value[keep] for key,value in g.items()}
    ikeep = (ioc['t']>=30)&(ioc['t']<TRAIN_END)&np.isfinite(ioc['offset'])&np.isfinite(ioc['spread'])
    ifit = {key:value[ikeep] for key,value in ioc.items()}
    np.savez_compressed(OUT/'training_marks.npz',**{'g_'+k:v for k,v in gfit.items()},
                        **{'ioc_'+k:v for k,v in ifit.items()})
    sys.path.insert(0,str(HERE.parent))
    from aggressive_size_fits import lognormal_fit
    ln = lognormal_fit(ifit['size'].astype(int),offset=.5)
    params = dict(source=str(source),train_window_s=[30,TRAIN_END],dev_window_s=[TRAIN_END,DEV_END],
                  test_window_s=[DEV_END,END],duration_s=END,
                  lambda_G=float(((g['t']>=30)&(g['t']<TRAIN_END)).sum()/(TRAIN_END-30)),
                  lambda_IOC=float(((ioc['t']>=30)&(ioc['t']<TRAIN_END)).sum()/(TRAIN_END-30)),
                  p_buy_G=float((gfit['side']==0).mean()),p_buy_IOC=float((ifit['side']==0).mean()),
                  ioc_mu=ln['mu'],ioc_sigma=ln['sigma'],initial_mid=10000,
                  training_limit_orders=len(gfit['t']),training_ioc_orders=len(ifit['t']),
                  limit_size_mean=float(gfit['size'].mean()),life_mean_s=float(gfit['life'].mean()),
                  limit_offset_quantiles=np.quantile(gfit['offset'],[0,.1,.5,.9,.99,1]).tolist(),
                  ioc_offset_quantiles=np.quantile(ifit['offset'],[0,.1,.5,.9,.99,1]).tolist(),
                  life_quantiles_s=np.quantile(gfit['life'],[0,.1,.5,.9,.99,1]).tolist())
    data = rec.data()
    summaries = {'train':summarize(data,30,TRAIN_END),'dev':summarize(data,TRAIN_END,DEV_END),
                 'test':summarize(data,DEV_END,END)}
    (OUT/'parameters.json').write_text(json.dumps(params,indent=2))
    (OUT/'original_summaries.json').write_text(json.dumps(summaries,indent=2))
    print('PARAMETERS',json.dumps(params),flush=True)
    print('ORIGINAL_SUMMARIES',json.dumps(summaries),flush=True)


if __name__=='__main__':
    main()
