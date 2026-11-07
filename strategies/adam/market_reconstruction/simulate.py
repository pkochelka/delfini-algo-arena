"""Fresh conditional marked-Poisson tapes; no historical paths or event times replayed."""
import argparse
import csv
import heapq
import json
from pathlib import Path
import time

import numpy as np

from engine import Book,Recorder,summarize

HERE = Path(__file__).resolve().parent
OUT = HERE/'results'


def state_code(spread,share):
    return min(5,max(1,int(spread)))*3+min(2,max(0,int(share*3)))


class Model:
    def __init__(self,conditioned=True):
        self.params = json.loads((OUT/'parameters.json').read_text())
        data = np.load(OUT/'training_marks.npz')
        self.g_marks = np.column_stack([data['g_size'],data['g_offset'],data['g_life']])
        self.g_pools = {}
        keys = np.minimum(5,np.maximum(1,data['g_spread'].astype(int)))*3+np.minimum(2,np.maximum(0,(data['g_share']*3).astype(int)))
        for key in np.unique(keys):
            indices = np.flatnonzero(keys==key)
            if len(indices)>=500:
                self.g_pools[int(key)] = indices
        self.ioc_offsets = data['ioc_offset']
        self.ioc_pools = {}
        bins = np.digitize(data['ioc_size'],[5,20,100,500])
        for key in range(5):
            self.ioc_pools[key] = np.flatnonzero(bins==key)
        self.conditioned = conditioned

    def g_mark(self,rng,spread,share):
        pool = self.g_pools.get(state_code(spread,share)) if self.conditioned else None
        index = int(pool[rng.integers(len(pool))]) if pool is not None else int(rng.integers(len(self.g_marks)))
        vol,offset,life = self.g_marks[index]
        return int(vol),int(offset),float(life)

    def ioc_mark(self,rng):
        p = self.params
        size = max(1,int(np.floor(rng.lognormal(p['ioc_mu'],p['ioc_sigma'])+.5)))
        group = int(np.digitize(size,[5,20,100,500]))
        pool = self.ioc_pools[group]
        offset = int(self.ioc_offsets[pool[rng.integers(len(pool))]])
        return size,offset


def generate(model,seed,duration,path=None,record_path=None):
    rng = np.random.default_rng(seed)
    p = model.params
    book,rec = Book(),Recorder(duration)
    cancellations = []
    next_g = rng.exponential(1/p['lambda_G'])
    next_ioc = rng.exponential(1/p['lambda_IOC'])
    oid = 1_000_000
    last_mid = p['initial_mid']
    last_us = -1
    handle = path.open('w',newline='') if path else None
    writer = csv.writer(handle,lineterminator='\n') if handle else None
    if writer:
        writer.writerow(['Time','Operation','OrderId','Side','Volume','Price','Lifespan'])
    started = time.perf_counter()
    try:
        while True:
            next_cancel = cancellations[0][0] if cancellations else float('inf')
            raw_time = min(next_g,next_ioc,next_cancel)
            if raw_time>=duration:
                break
            # Distinct microsecond timestamps, matching the observed encoding convention.
            t_us = max(last_us+1,int(round(raw_time*1e6)))
            t = t_us/1e6
            if t>=duration:
                break
            last_us = t_us
            rec.advance(t,book)
            if raw_time==next_cancel:
                _,old_id = heapq.heappop(cancellations)
                rec.cancel(t,old_id,book)
                if writer:
                    writer.writerow([f'{t:.6f}','Cancel',old_id,'','','',''])
                continue
            is_ioc = raw_time==next_ioc
            if is_ioc:
                next_ioc += rng.exponential(1/p['lambda_IOC'])
            else:
                next_g += rng.exponential(1/p['lambda_G'])
            oid += 1
            side = 0 if rng.random() < p['p_buy_IOC' if is_ioc else 'p_buy_G'] else 1
            b,a = book.best(0),book.best(1)
            if b and a:
                last_mid = (b+a)/2
            spread = a-b if a and b else 1
            total = book.qty(0)+book.qty(1)
            share = book.qty(side)/total if total else .5
            if is_ioc:
                vol,offset = model.ioc_mark(rng)
                opposite = book.best(1-side)
                if not opposite:
                    opposite = int(np.ceil(last_mid+.5)) if side==0 else int(np.floor(last_mid-.5))
                price = opposite+offset*(1 if side==0 else -1)
            else:
                vol,offset,life = model.g_mark(rng,spread,share)
                own = book.best(side)
                if not own:
                    own = int(np.floor(last_mid-.5)) if side==0 else int(np.ceil(last_mid+.5))
                price = own-offset*(1 if side==0 else -1)
                opposite = book.best(1-side)
                if opposite:
                    price = min(price,opposite-1) if side==0 else max(price,opposite+1)
                heapq.heappush(cancellations,(t+life,oid))
            price = max(1,int(price))
            result = rec.insert(t,oid,side,price,vol,is_ioc,book)
            assert is_ioc or result[0]==0
            if writer:
                writer.writerow([f'{t:.6f}','Insert',oid,'B' if side==0 else 'A',vol,price,'F' if is_ioc else 'G'])
        rec.advance(duration,book)
    finally:
        if handle:
            handle.close()
    if record_path:
        rec.save(record_path)
    summary = summarize(rec.data(),30,duration)
    summary.update(seed=seed,conditioned=model.conditioned,runtime_s=round(time.perf_counter()-started,2))
    return summary,rec


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=20261107)
    parser.add_argument('--duration',type=float,default=2430)
    parser.add_argument('--name',default='pilot_conditional')
    parser.add_argument('--unconditional',action='store_true')
    parser.add_argument('--write-csv',action='store_true')
    args = parser.parse_args()
    m = Model(not args.unconditional)
    result,_ = generate(m,args.seed,args.duration,OUT/(args.name+'.csv') if args.write_csv else None,OUT/(args.name+'.npz'))
    (OUT/(args.name+'.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
