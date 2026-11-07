"""Unretuned Round 1 strategies, seeded Round 2 validation/test replays."""
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime,timezone
from extract import HERE,REPO,SOURCE,GEN,OUT,dump

sys.path.insert(0,str(HERE.parent/'ladder_research'))
from research import AuditSimulator
from simulator.simulator import ExchangeSimulator,SimulationConfig

class StreamingAudit(AuditSimulator):
    iter_backtest=staticmethod(ExchangeSimulator.iter_backtest)
    count_backtest_rows=staticmethod(ExchangeSimulator.count_backtest_rows)

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def split():
    peter=module(REPO/'strategies'/'peter'/'split_data.py','peter_splitter')
    windows=[dict(name='validation',start=3540.,trade=3600.,end=4790.),dict(name='test',start=4740.,trade=4800.,end=5990.)]
    book=peter.Book();active={}
    try:
        with SOURCE.open(newline='') as f:
            r=csv.reader(f);next(r)
            for row in r:
                t=float(row[0])
                for w in windows:
                    name=w['name'];path=GEN/(name+'.csv')
                    if name not in active and t>=w['start']:
                        peter.write_window(path,[],[],w['start'],w['end'],book.snapshot())
                        handle=path.open('a',newline='');active[name]=(handle,csv.writer(handle,lineterminator='\n'))
                    if name in active and w['start']<=t<w['end']:active[name][1].writerow(row)
                if t>=5990:break
                book.apply(row)
    finally:
        for f,_ in active.values():f.close()
    return windows

def run(task):
    name,path,params,w=task
    m=module(Path(path),'baseline_strategy')
    m.PARAMS.update(dict(params,trade_start_us=round(w['trade']*1e6)))
    strategy=m.AdaptiveSpreadStrategy() if name=='adaptive_spread' else m.LadderMM()
    logs=GEN/f'{name}_{w["name"]}_logs'
    logs.mkdir(exist_ok=True)
    cfg=SimulationConfig(order_accept_delay_ms=5,max_position=100,max_potential_position=1000,
        position_breach_penalty_per_second=10000,run_log_path=logs/'run.jsonl',
        book_delta_log_path=logs/'book_deltas.csv',event_log_path=logs/'events.csv',trade_log_path=logs/'trades.csv')
    sim=StreamingAudit(strategy,cfg)
    result=sim.run(GEN/(w['name']+'.csv'))
    assert sim.max_inventory<=strategy.p.get('cap',80) and result.breach_penalty==0
    assert sim.first_submit_us>=round(w['trade']*1e6)
    r=dict(model=name,split=w['name'],max_abs_inventory=sim.max_inventory,
        duration_s=w['end']-w['trade'],pnl_per_hour=result.pnl/(w['end']-w['trade'])*3600,
        **result.model_dump(mode='json'))
    (logs/'result.json').write_text(json.dumps(result.model_dump(mode='json'),indent=2))
    (logs/'metadata.json').write_text(json.dumps(dict(trader=name,trader_class=strategy.__class__.__name__,
        data_file=str(GEN/(w['name']+'.csv')),latency=.005,position_limit=100,
        max_potential_position=1000,position_breach_penalty_per_second=10000,
        run_timestamp=datetime.now(timezone.utc).isoformat()),indent=2))
    curve=sim.pnl_curve(result.pnl)
    with (logs/'pnl.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['elapsed_s','pnl']);wr.writerows(((t-curve[0][0])/1e6,p) for t,p in curve)
    return r

def main():
    snapshots=GEN/'snapshots';snapshots.mkdir(exist_ok=True)
    paths={}
    for name,source in [('ladder',REPO/'strategies'/'peter'/'ladder_mm.py'),('adaptive_spread',HERE.parent/'models'/'adaptive_spread.py')]:
        dest=snapshots/(name+'.py');shutil.copyfile(source,dest);paths[name]=str(dest)
    tuned=json.loads((HERE.parent/'ladder_research'/'results'/'locked_selection.json').read_text())['overrides']
    candidates=[('ladder_native',paths['ladder'],{}),('ladder_tuned_round1',paths['ladder'],tuned),('adaptive_spread',paths['adaptive_spread'],{})]
    dump('baseline_protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),candidates=candidates,
        hashes={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in paths.items()},
        selection='None; fixed Round1 settings for transfer diagnosis only. Model predictions not applied to these strategies.'))
    windows=split();results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        tasks=[(name,path,p,w) for name,path,p in candidates for w in windows]
        for r in pool.map(run,tasks):
            results.append(r);print(json.dumps(r),flush=True)
    dump('baseline_results.json',results)
    print('BASELINES COMPLETE',flush=True)

if __name__=='__main__':main()
