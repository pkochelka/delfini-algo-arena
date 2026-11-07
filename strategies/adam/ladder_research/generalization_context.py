"""Compare the observed market regimes without changing or caching repo files."""
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from research import REPO,SPLITS
from analysis.replay import replay_csv
from analysis.cache import load_tape

ROOT=HERE/'generalization'
records=[]
for name,path in [('round1_test',SPLITS/'test.csv'),('legacy_sample',REPO/'data'/'sample.csv')]:
    tape=load_tape(path) if name=='round1_test' else replay_csv(path)
    q=tape.quotes
    valid=q.dropna(subset=['mid','spread'])
    qt=valid['t_us'].to_numpy()
    grid=np.arange(tape.start_us+60_050_000,tape.end_us,100_000)
    ix=np.maximum(0,np.searchsorted(qt,grid,side='right')-1)
    mid=valid['mid'].to_numpy()[ix]
    spread=valid['spread'].to_numpy()[ix]
    touch=(valid['bid_qty'].to_numpy()[ix]+valid['ask_qty'].to_numpy()[ix])/2
    orders=tape.orders
    ioc=orders[(orders['ioc']==1)&(orders['t_us']>=grid[0])]
    duration=(tape.end_us-grid[0])/1e6
    record=dict(name=name,duration_s=duration,mean_spread=float(np.mean(spread)),
        one_tick_spread_fraction=float(np.mean(spread==1)),mean_touch_quantity=float(np.mean(touch)),
        sigma_1s_ticks=float(np.std(np.diff(mid[::10]))),mid_change=float(mid[-1]-mid[0]),
        ioc_per_second=len(ioc)/duration,mean_ioc_size=float(ioc['volume'].mean()),
        ioc_size_q99=float(ioc['volume'].quantile(.99)))
    records.append(record)
    print(json.dumps(record),flush=True)
(ROOT/'market_regimes.json').write_text(json.dumps(records,indent=2))
