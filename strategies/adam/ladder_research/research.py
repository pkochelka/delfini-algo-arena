"""LadderMM: identical split/search protocol to adaptive_spread, without source edits."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').exists() and (p / 'simulator').is_dir())
SOURCE = REPO/'strategies'/'peter'/'ladder_mm.py'
SPLITS = HERE.parent/'spread_research'/'splits'
OUT = HERE/'results'
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(REPO))
from simulator.simulator import ExchangeSimulator,SimulationConfig
from simulator.models import OrderStatus

EVENT_CACHE = {}


def load(overrides=None):
    spec = importlib.util.spec_from_file_location('ladder_trial_module',SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PARAMS.update(overrides or {})
    return module.LadderMM()


class AuditSimulator(ExchangeSimulator):
    def __init__(self,*args,**kwargs):
        self.max_inventory = 0
        self.order_rejects = 0
        self.cancel_rejects = 0
        self.max_order_quantity = 0
        self.first_submit_us = None
        self.order_count = 0
        super().__init__(*args,**kwargs)

    def iter_backtest(self,path):
        key = str(path)
        if key not in EVENT_CACHE:
            EVENT_CACHE[key] = list(ExchangeSimulator.iter_backtest(path))
        return iter(EVENT_CACHE[key])

    def count_backtest_rows(self,path):
        self.iter_backtest(path)
        return len(EVENT_CACHE[str(path)])

    def submit_order(self,request):
        self.max_order_quantity = max(self.max_order_quantity,request.quantity)
        self.order_count += 1
        if self.first_submit_us is None:
            self.first_submit_us = self.current_ts_us
        return super().submit_order(request)

    def _private_notify(self,event):
        self.max_inventory = max(self.max_inventory,abs(self._inventory))
        if event.status == OrderStatus.REJECTED:
            if event.message and event.message.startswith('cancel rejected'):
                self.cancel_rejects += 1
            else:
                self.order_rejects += 1
        super()._private_notify(event)


def run(task):
    label,overrides,window,full = task
    strategy = load(dict(overrides,trade_start_us=window['trade_start_us']))
    cap = strategy.p['cap']
    logs = OUT/('full_logs_'+label) if full else None
    if logs:
        logs.mkdir(exist_ok=True)
    config = SimulationConfig(order_accept_delay_ms=5,max_position=100,max_potential_position=1000,
                              position_breach_penalty_per_second=10_000,
                              run_log_path=logs/'run.jsonl' if logs else None,
                              book_delta_log_path=logs/'book_deltas.csv' if logs else None,
                              event_log_path=logs/'events.csv' if logs else None,
                              trade_log_path=logs/'trades.csv' if logs else None)
    started = time.perf_counter()
    sim = AuditSimulator(strategy,config)
    result = sim.run(Path(window['file']))
    cap_ok = sim.max_inventory<=cap and result.breach_penalty==0
    warmup_ok = sim.first_submit_us is None or sim.first_submit_us>=window['trade_start_us']
    size_ok = label=='peter_original' or sim.max_order_quantity<=10
    if not cap_ok or not warmup_ok or not size_ok:
        raise AssertionError(dict(label=label,cap_ok=cap_ok,warmup_ok=warmup_ok,size_ok=size_ok,
                                 max_inventory=sim.max_inventory,max_order_quantity=sim.max_order_quantity))
    curve = sim.pnl_curve(result.pnl)
    running_peak = 0.0
    drawdown = 0.0
    for _,pnl in curve:
        running_peak = max(running_peak,pnl)
        drawdown = max(drawdown,running_peak-pnl)
    record = dict(label=label,window=window['name'],overrides=overrides,effective_params=dict(strategy.p),
                  seconds=round(time.perf_counter()-started,3),max_abs_inventory=sim.max_inventory,
                  first_submit_us=sim.first_submit_us,max_order_quantity=sim.max_order_quantity,
                  submitted_orders=sim.order_count,order_rejections=sim.order_rejects,
                  cancel_race_rejections=sim.cancel_rejects,max_drawdown=drawdown,
                  **result.model_dump(mode='json'))
    (OUT/(label+'_'+window['name']+'.json')).write_text(json.dumps(record,indent=2))
    (OUT/(label+'_'+window['name']+'_pnl.json')).write_text(json.dumps(
        [(round((t-window['trade_start_us'])/1e6,6),p) for t,p in curve]))
    if logs:
        import csv
        (logs/'result.json').write_text(json.dumps(result.model_dump(mode='json'),indent=2))
        metadata = dict(trader='ladder comparison',trader_class='LadderMM',data_file=window['file'],
                        latency=.005,position_limit=100,max_potential_position=1000,
                        position_breach_penalty_per_second=10_000,run_timestamp=datetime.now(timezone.utc).isoformat())
        (logs/'metadata.json').write_text(json.dumps(metadata,indent=2))
        with (logs/'pnl.csv').open('w',newline='') as file:
            writer = csv.writer(file,lineterminator='\n')
            writer.writerow(['elapsed_s','pnl'])
            writer.writerows(((t-curve[0][0])/1e6,p) for t,p in curve)
    return record


def score(records):
    pnls = [v['pnl'] for v in records]
    return statistics.fmean(pnls)-.5*statistics.pstdev(pnls)-5*statistics.fmean(v['abs_inventory_q95'] for v in records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs',type=int,default=2)
    args = parser.parse_args()
    windows = {v['name']:v for v in json.loads((SPLITS/'folds.json').read_text())['windows']}
    previous = json.loads((HERE.parent/'spread_research'/'results'/'protocol.json').read_text())
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    candidates = {}
    for levels in (1,2):
        for offset in (0,1):
            for topup in (0,5):
                for skew in (0,40):
                    label = f'L{levels}_o{offset}_t{topup}_s{skew}'
                    candidates[label] = dict(cap=80,size_top=10,size_deep=10 if levels==2 else 0,
                                             levels=levels,tight_offset=offset,topup=topup,skew_lots=skew,
                                             improve_reduce=0,flat_at=0,flat_to=0)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),strategy=str(SOURCE),source_sha256=source_hash,
                    candidates=candidates,screen_folds=previous['screen_folds'],fit_folds=previous['fit_folds'],
                    validation_folds=previous['validation_folds'],objective=previous['objective'],
                    selection=previous['selection'],test='test',matched_cap=80,matched_max_order_size=10,
                    original_default_test=True,other_benchmark='same source defaults with cap=80,size_top=10,size_deep=0',
                    test_already_seen='This is the identical test previously evaluated for adaptive_spread; excluded from ladder tuning, not a fresh blind dataset.',
                    split_sha256={key:hashlib.sha256(Path(v['file']).read_bytes()).hexdigest() for key,v in windows.items()})
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    cache = {}
    def collect(pool,label,overrides,folds,full=False):
        futures = []
        for name in folds:
            key = (label,name)
            if key in cache:
                continue
            existing = OUT/(label+'_'+name+'.json')
            if existing.exists():
                record = json.loads(existing.read_text())
                if record['overrides']==overrides:
                    cache[key] = record
                    print('RESUME',label,name,record['pnl'],flush=True)
                    continue
            futures.append(pool.submit(run,(label,overrides,windows[name],full)))
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            cache[(record['label'],record['window'])] = record
            print(json.dumps({k:record[k] for k in ('label','window','pnl','traded_volume','max_abs_inventory','seconds')}),flush=True)
        return [cache[(label,name)] for name in folds]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        ranking = []
        for name,overrides in candidates.items():
            records = collect(pool,name,overrides,protocol['screen_folds'])
            ranking.append((score(records),name))
        finalists = [name for _,name in sorted(ranking,reverse=True)[:3]]
        training,validation = {},{}
        print('FINALISTS',finalists,flush=True)
        for name in finalists:
            training[name] = collect(pool,name,candidates[name],protocol['fit_folds'])
            validation[name] = collect(pool,name,candidates[name],protocol['validation_folds'])
        winner = max(finalists,key=lambda name:score(validation[name]))
        locked = dict(label=winner,overrides=candidates[winner],locked_utc=datetime.now(timezone.utc).isoformat(),
                      training_scores={n:score(v) for n,v in training.items()},
                      validation_scores={n:score(v) for n,v in validation.items()},test_not_yet_evaluated=True)
        (OUT/'locked_selection.json').write_text(json.dumps(locked,indent=2))
        print('LOCKED',json.dumps(locked),flush=True)
        chosen = collect(pool,winner,candidates[winner],['test'],full=True)[0]
        defaults = collect(pool,'peter_original',{},['test'],full=True)[0]
        matched = collect(pool,'peter_default_matched',dict(cap=80,size_top=10,size_deep=0),['test'])[0]
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==source_hash,'Original strategy changed during study'
        outcome = dict(protocol=protocol,screen_ranking=sorted(ranking,reverse=True),locked=locked,
                       training=training,validation=validation,test=chosen,original_defaults=defaults,
                       matched_defaults=matched,original_source_unchanged=True)
        (OUT/'outcome.json').write_text(json.dumps(outcome,indent=2))
        print('OUTCOME',json.dumps(dict(test=chosen,original_defaults=defaults,matched_defaults=matched)),flush=True)


if __name__=='__main__':
    main()
