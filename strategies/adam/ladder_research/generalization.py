"""Frozen-strategy temporal, parameter, cross-tape and structural robustness audit."""
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE/'generalization'
ROOT.mkdir(exist_ok=True)
sys.path.insert(0,str(HERE))
from research import REPO,SOURCE,SPLITS,AuditSimulator,ExchangeSimulator,SimulationConfig
import numpy as np

FROZEN=ROOT/'ladder_frozen.py'
TUNED=json.loads((HERE/'results'/'locked_selection.json').read_text())['overrides']


class StreamingAudit(AuditSimulator):
    iter_backtest=staticmethod(ExchangeSimulator.iter_backtest)
    count_backtest_rows=staticmethod(ExchangeSimulator.count_backtest_rows)


def load(params):
    spec=importlib.util.spec_from_file_location('frozen_ladder',FROZEN)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.PARAMS.update(params)
    return m.LadderMM()


def run(task):
    label,params,w=task
    target=ROOT/(label+'__'+w['name']+'.json')
    if target.exists():
        return json.loads(target.read_text())
    s=load(dict(params,trade_start_us=w['trade_start_us']))
    cfg=SimulationConfig(order_accept_delay_ms=5,max_position=100,max_potential_position=1000,
                         position_breach_penalty_per_second=10000)
    start=time.perf_counter()
    sim=StreamingAudit(s,cfg)
    result=sim.run(Path(w['file']))
    assert result.breach_penalty==0 and sim.max_inventory<=s.p['cap']
    assert sim.first_submit_us is None or sim.first_submit_us>=w['trade_start_us']
    curve=sim.pnl_curve(result.pnl)
    x=np.array([t for t,p in curve],dtype=float)
    y=np.array([p for t,p in curve],dtype=float)
    duration=(w['end_s']*1e6-w['trade_start_us'])/1e6
    minute_grid=np.arange(w['trade_start_us'],w['end_s']*1e6+1,60e6)
    minute_changes=np.diff(np.interp(minute_grid,x,y))
    # Conditional uncertainty within this tape, not across market regimes.
    ci=None
    if len(minute_changes)>=10:
        rng=np.random.default_rng(20261107)
        n=len(minute_changes)
        starts=rng.integers(n,size=(3000,math.ceil(n/5)))
        indices=((starts[:,:,None]+np.arange(5))%n).reshape(3000,-1)[:,:n]
        rates=minute_changes[indices].mean(axis=1)*60
        ci=np.quantile(rates,[.025,.975]).tolist()
    hist=sim._abs_inventory_time_us
    record=dict(label=label,window=w['name'],params=dict(s.p),duration_s=duration,
                pnl_per_hour=result.pnl/duration*3600,max_abs_inventory=sim.max_inventory,
                max_drawdown=float(np.max(np.maximum.accumulate(np.r_[0,y])[1:]-y)),
                time_at_limit_fraction=sum(v for k,v in hist.items() if k>=100)/1e6/duration,
                block_bootstrap_95_pnl_per_hour=ci,minute_changes=minute_changes.tolist(),
                order_rejections=sim.order_rejects,seconds=round(time.perf_counter()-start,2),
                **result.model_dump(mode='json'))
    target.write_text(json.dumps(record,indent=2))
    return record


def generate(seed):
    path=ROOT/f'synthetic_{seed}.csv'
    if not path.exists():
        sys.path.insert(0,str(HERE.parent/'market_reconstruction'))
        from simulate import Model,generate as make
        summary,_=make(Model(True),seed,1260,path=path)
        path.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2))
    return dict(name=f'synthetic_{seed}',file=str(path),trade_start_us=60_000_000,end_s=1260)


def stresses(window):
    kinds=['ioc_half','ioc_double','ioc_cap500','drop_half_buy_ioc','drop_half_sell_ioc']
    paths={k:ROOT/(k+'.csv') for k in kinds}
    if not all(p.exists() for p in paths.values()):
        handles={k:p.open('w',newline='') for k,p in paths.items()}
        writers={k:csv.writer(f,lineterminator='\n') for k,f in handles.items()}
        rng=np.random.default_rng(77219)
        try:
            with Path(window['file']).open(newline='') as f:
                reader=csv.reader(f)
                head=next(reader)
                for w in writers.values():w.writerow(head)
                for r in reader:
                    ioc=r[1]=='Insert' and r[6]=='F'
                    u=rng.random() if ioc else 1
                    for k,w in writers.items():
                        if ioc and u<.5 and ((k=='drop_half_buy_ioc' and r[3]=='B') or (k=='drop_half_sell_ioc' and r[3]=='A')):
                            continue
                        row=r.copy()
                        if ioc:
                            q=int(row[4])
                            if k=='ioc_half':row[4]=max(1,round(q*.5))
                            if k=='ioc_double':row[4]=q*2
                            if k=='ioc_cap500':row[4]=min(q,500)
                        w.writerow(row)
        finally:
            for f in handles.values():f.close()
    return [dict(window,name=k,file=str(paths[k])) for k in kinds]


def main():
    if not FROZEN.exists():shutil.copyfile(SOURCE,FROZEN)
    sha=hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    assert sha==hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'Existing frozen source differs: do not mix versions.'
    windows=json.loads((SPLITS/'folds.json').read_text())['windows']
    sample=dict(name='legacy_sample',file=str(REPO/'data'/'sample.csv'),trade_start_us=60_000_000,end_s=3599.000003)
    neighbors={
        'size100':{'size_top':100},'size150':{'size_top':150},'size200':{'size_top':200},
        'topup0':{'topup':0},'topup5':{'topup':5},'topup20':{'topup':20},'topup40':{'topup':40}}
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=str(SOURCE),source_sha256=sha,
        frozen_source=str(FROZEN),temporal_windows=windows,legacy_sample=sample,
        models={'native':{},'tuned':TUNED},neighbors=neighbors,neighbor_windows=['wf05','wf06'],
        source_version_control='Also run size_top=200 on test to reconcile the prior native result; 400 may replenish tiny residuals differently.',
        synthetic_seeds=[7101,7102,7103,7104,7105],synthetic_duration_s=1260,warmup_s=60,
        stresses=['halve IOC sizes','double IOC sizes','cap IOC sizes at 500','drop half buy IOC messages','drop half sell IOC messages'],
        stress_base='previously evaluated test slice',retuning=False,
        interpretation='All real tapes previously available/partly inspected. Synthetic generator calibrated to round1 has known optimistic markouts and too-low volatility. Stress tapes reuse source limit messages and prices and are counterfactuals, not independent samples. No strategy selected on these results.')
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print('FROZEN',sha,flush=True)
    tasks=[]
    for w in windows+[sample]:
        for label,p in (('native',{}),('tuned',TUNED)):
            tasks.append((label,p,w))
    for label,p in neighbors.items():
        for w in windows:
            if w['name'] in ('wf05','wf06'):tasks.append((label,p,w))
    tasks.append(('size200',{'size_top':200},next(w for w in windows if w['name']=='test')))
    stress_windows=stresses(next(w for w in windows if w['name']=='test'))
    for w in stress_windows:
        for label,p in (('native',{}),('tuned',TUNED)):tasks.append((label,p,w))
    records=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for w in pool.map(generate,protocol['synthetic_seeds']):
            print('GENERATED',w['name'],flush=True)
            for label,p in (('native',{}),('tuned',TUNED)):tasks.append((label,p,w))
        futures=[pool.submit(run,t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            r=future.result()
            records.append(r)
            print(json.dumps({k:r[k] for k in ['label','window','pnl','pnl_per_hour','max_abs_inventory','seconds']}),flush=True)
    native_test=next(r for r in records if r['label']=='native' and r['window']=='test')
    tuned_test=next(r for r in records if r['label']=='tuned' and r['window']=='test')
    old_native=next(r for r in records if r['label']=='size200' and r['window']=='test')
    assert old_native['pnl']==107532 and tuned_test['pnl']==22254,'Prior settings failed to reproduce: investigate.'
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==sha
    (ROOT/'outcome.json').write_text(json.dumps(dict(protocol=protocol,records=records,original_unchanged=True,
        prior_native_pnl_exact_match_at_size200=True,prior_tuned_pnl_exact_match=True,
        current_native_pnl_change=native_test['pnl']-old_native['pnl']),indent=2))
    print('COMPLETE',len(records),'runs; prior size200/tuned settings reproduce previous test exactly.',flush=True)


if __name__=='__main__':main()
