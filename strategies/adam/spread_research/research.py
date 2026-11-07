"""Fixed-protocol chronological tuning and one locked test of adaptive_spread.py."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').exists() and (p / 'simulator').is_dir())
sys.path.insert(0,str(REPO))
from simulator.simulator import ExchangeSimulator, SimulationConfig

STRATEGY = REPO/'strategies'/'adam'/'models'/'adaptive_spread.py'
RESULTS = HERE/'results'
RESULTS.mkdir(exist_ok=True)
EVENT_CACHE = {}


class AuditSimulator(ExchangeSimulator):
    def __init__(self,*args,**kwargs):
        self.max_inventory = 0
        self.half_samples = []
        self.next_sample = 0
        self.order_rejects = 0
        self.cancel_race_rejects = 0
        super().__init__(*args,**kwargs)

    def iter_backtest(self,path):
        key = str(path)
        if key not in EVENT_CACHE:
            EVENT_CACHE[key] = list(ExchangeSimulator.iter_backtest(path))
        return iter(EVENT_CACHE[key])

    def count_backtest_rows(self,path):
        self.iter_backtest(path)
        return len(EVENT_CACHE[str(path)])

    def _private_notify(self,event):
        self.max_inventory = max(self.max_inventory,abs(self._inventory))
        if event.status.value=='rejected':
            if event.message and event.message.startswith('cancel rejected'):
                self.cancel_race_rejects += 1
            else:
                self.order_rejects += 1
        super()._private_notify(event)

    def _emit_public_snapshot(self):
        super()._emit_public_snapshot()
        strategy = self.strategy
        if self.current_ts_us>=self.next_sample and self.current_ts_us>=strategy.p['trade_start_us']:
            self.half_samples.append((self.current_ts_us,strategy.last_half_spread,strategy.variance,strategy.toxicity))
            self.next_sample = self.current_ts_us+1_000_000


def load(overrides):
    spec = importlib.util.spec_from_file_location('spread_strategy_trial',STRATEGY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PARAMS.update(overrides)
    return module.AdaptiveSpreadStrategy()


def run(task):
    label,overrides,window,full = task
    overrides = dict(overrides,trade_start_us=window['trade_start_us'])
    strategy = load(overrides)
    full_dir = RESULTS/'test_full_logs' if full else None
    if full_dir:
        full_dir.mkdir(exist_ok=True)
    config = SimulationConfig(order_accept_delay_ms=5,max_position=100,max_potential_position=1000,
                              position_breach_penalty_per_second=10_000,
                              run_log_path=full_dir/'run.jsonl' if full else None,
                              book_delta_log_path=full_dir/'book_deltas.csv' if full else None,
                              event_log_path=full_dir/'events.csv' if full else None,
                              trade_log_path=full_dir/'trades.csv' if full else None)
    started = time.perf_counter()
    simulator = AuditSimulator(strategy,config)
    result = simulator.run(Path(window['file']))
    assert simulator.max_inventory<=strategy.p['position_limit']
    assert result.breach_penalty==0 and simulator.order_rejects==0
    half = [v[1] for v in simulator.half_samples]
    record = dict(label=label,window=window['name'],overrides=overrides,seconds=round(time.perf_counter()-started,3),
                  max_abs_inventory=simulator.max_inventory,order_rejections=simulator.order_rejects,
                  cancel_race_rejections=simulator.cancel_race_rejects,
                  half_spread_mean=statistics.fmean(half) if half else None,
                  half_spread_min=min(half) if half else None,half_spread_max=max(half) if half else None,
                  **result.model_dump(mode='json'))
    curve = simulator.pnl_curve(result.pnl)
    pnl_path = RESULTS/(label+'_'+window['name']+'_pnl.json')
    pnl_path.write_text(json.dumps([(round((t-window['trade_start_us'])/1e6,6),p) for t,p in curve]))
    (RESULTS/(label+'_'+window['name']+'_half_spread.json')).write_text(json.dumps(simulator.half_samples))
    if full:
        import csv
        (full_dir/'result.json').write_text(json.dumps(result.model_dump(mode='json'),indent=2))
        metadata = dict(trader='adaptive spread research',trader_class='AdaptiveSpreadStrategy',
                        data_file=window['file'],latency=.005,position_limit=100,max_potential_position=1000,
                        position_breach_penalty_per_second=10_000,run_timestamp=datetime.now().isoformat())
        (full_dir/'metadata.json').write_text(json.dumps(metadata,indent=2))
        with (full_dir/'pnl.csv').open('w',newline='') as file:
            writer = csv.writer(file,lineterminator='\n')
            writer.writerow(['elapsed_s','pnl'])
            writer.writerows(((t-curve[0][0])/1e6,p) for t,p in curve)
    (RESULTS/(label+'_'+window['name']+'.json')).write_text(json.dumps(record,indent=2))
    return record


def score(records):
    p = [r['pnl'] for r in records]
    return statistics.fmean(p)-.5*statistics.pstdev(p)-5*statistics.fmean(r['abs_inventory_q95'] for r in records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--pilot',action='store_true')
    args = parser.parse_args()
    metadata = json.loads((HERE/'splits'/'folds.json').read_text())
    windows = {d['name']:d for d in metadata['windows']}
    trials = {}
    for mult in (.5,1.,2.,3.):
        for half in (5.,30.):
            for tox in (0.,1.):
                name = f'm{mult:g}_v{half:g}_t{tox:g}'
                trials[name] = dict(spread_mult=mult,vol_halflife_s=half,tox_weight=tox)
    baselines = {'fixed_touch':dict(adaptive=False,fixed_half_spread=.5),
                 'fixed_wide':dict(adaptive=False,fixed_half_spread=1.5)}
    protocol = dict(created=datetime.now().isoformat(),source=str(STRATEGY),
                    initial_strategy_sha256=hashlib.sha256(STRATEGY.read_bytes()).hexdigest(),
                    candidates=trials,screen_folds=['wf01','wf03'],fit_folds=['wf01','wf02','wf03','wf04'],
                    validation_folds=['wf05','wf06'],test='test',
                    objective='mean(PnL) - 0.5 * population_std(PnL) - 5 * mean(q95_abs_inventory)',
                    selection='screen all candidates on wf01/wf03; top 3 on remaining fit folds; validation chooses among the 3; lock before test',
                    risk_limit=80,order_size=10,baselines=baselines)
    (RESULTS/'protocol.json').write_text(json.dumps(protocol,indent=2))
    cache = {}
    def collect(pool,label,overrides,fold_names,full=False):
        todo = []
        for fold in fold_names:
            key = (label,fold)
            if key in cache:
                continue
            existing = RESULTS/(label+'_'+fold+'.json')
            if existing.exists():
                record = json.loads(existing.read_text())
                expected = dict(overrides,trade_start_us=windows[fold]['trade_start_us'])
                if record['overrides']==expected:
                    cache[key] = record
                    print('RESUME',label,fold,record['pnl'],flush=True)
                    continue
            todo.append(pool.submit(run,(label,overrides,windows[fold],full)))
        for future in concurrent.futures.as_completed(todo):
            record = future.result()
            cache[(record['label'],record['window'])] = record
            print(json.dumps({k:record[k] for k in ('label','window','pnl','traded_volume','max_abs_inventory','seconds')}),flush=True)
        return [cache[(label,fold)] for fold in fold_names]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        if args.pilot:
            collect(pool,'pilot_adaptive',trials['m1_v5_t1'],['wf01'])
            collect(pool,'pilot_fixed',baselines['fixed_touch'],['wf01'])
            return
        ranking = []
        for label,overrides in trials.items():
            records = collect(pool,label,overrides,protocol['screen_folds'])
            ranking.append((score(records),label))
        finalists = [name for _,name in sorted(ranking,reverse=True)[:3]]
        print('FINALISTS',finalists,flush=True)
        training = {}
        validation = {}
        for label in finalists:
            training[label] = collect(pool,label,trials[label],protocol['fit_folds'])
            validation[label] = collect(pool,label,trials[label],protocol['validation_folds'])
        winner = max(finalists,key=lambda label:score(validation[label]))
        locked = dict(label=winner,overrides=trials[winner],locked_at=datetime.now().isoformat(),
                      test_not_yet_evaluated=True,training_scores={n:score(v) for n,v in training.items()},
                      validation_scores={n:score(v) for n,v in validation.items()})
        (RESULTS/'locked_selection.json').write_text(json.dumps(locked,indent=2))
        print('LOCKED',json.dumps(locked),flush=True)
        chosen = collect(pool,winner,trials[winner],['test'],full=True)[0]
        benchmark = {}
        for label,overrides in baselines.items():
            benchmark[label] = collect(pool,label,overrides,['test'])[0]
        outcome = dict(protocol=protocol,screen_ranking=sorted(ranking,reverse=True),locked=locked,
                       training=training,validation=validation,test=chosen,baselines=benchmark)
        (RESULTS/'outcome.json').write_text(json.dumps(outcome,indent=2))
        print('FINAL_OUTCOME',json.dumps(dict(winner=winner,test=chosen,baselines=benchmark)),flush=True)


if __name__=='__main__':
    main()
