"""Offline replay of FV (from a strategy file) on a synth tape vs true S. Emulates the strategy's drought/hot logic.
uv run python off.py STRAT.py SEED [--set k=v ...] [--t1 6000]"""
import sys, json, math, argparse, ast, importlib.util, itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from analysis.cache import load_tape

def load_mod(path):
    spec = importlib.util.spec_from_file_location('m', path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def run(strat, seed, over, t1=6000.0):
    m = load_mod(strat)
    p = dict(m.PARAMS) | over
    tp = load_tape(str(ROOT / f'data/synth/r2_v2_seed{seed}.csv'), verbose=False)
    ev, o = tp.events, tp.orders
    filled = np.zeros(len(ev), dtype=np.int64)
    oo = o[o.ioc]; filled[oo.row.to_numpy()] = oo.filled.to_numpy()
    bid = ev.bid.to_numpy(); ask = ev.ask.to_numpy()
    T = ev.t_us.to_numpy() / 1e6; ins = (ev.op == 'I').to_numpy(); G = (ev.lifespan == 'G').to_numpy()
    B = (ev.side == 'B').to_numpy(); px = np.nan_to_num(ev.price.to_numpy()).astype(np.int64)
    pb = np.r_[0, bid[:-1]]; pa = np.r_[0, ask[:-1]]
    SP = np.load(ROOT / f'marek/research/round2/hyp/filter/S_seed{seed}.npy')
    tr = json.load(open(ROOT / f'data/synth/r2_v2_seed{seed}.truth.json'))
    f = m.FV(p, (pb[1] + pa[1]) / 2 if pb[1] and pa[1] else 10000)
    from collections import deque
    it = deque(); sg_t = None; sg_next = 0; sg_j = 0
    samp = []; nxt = 20.0
    for i in range(1, len(T)):
        t = T[i]
        if t > t1: break
        if not ins[i]: continue
        if G[i]:
            opp = pa[i] if B[i] else pb[i]
            f.maker(t, bool(B[i]), int(px[i]), int(opp) if opp else None)
            it.append(t)
        else:
            if filled[i]:
                f.taker(t, bool(B[i]), None, int(filled[i]))
        # drought logic (strategy _shock)
        while it and it[0] < t - p['sg_win']: it.popleft()
        if sg_t is None:
            if t > p['sg_win'] and len(it) < p['sg_thr'] and t > sg_next:
                sg_t = t; sg_j = f.jumps; sg_next = t + p['shock_cool']; f.drought = True
        if sg_t is not None:
            rel = t - sg_t
            if (f.jumps != sg_j and rel > p['shock_lead'] - 0.5) or rel > p['shock_lead'] + p['shock_len']:
                sg_t = None; f.drought = f.hot = False
            else:
                f.hot = rel >= p['shock_lead']
        if t >= nxt:
            nxt = t + 0.05
            samp.append((t, f.mean, f.sd, f.pv, f.exc))
    S = np.array(samp)
    j = np.searchsorted(SP[:, 0], S[:, 0], 'right') - 1
    return S, SP[j, 1], tr

def score(S, Strue, tr):
    t, F, sd, pv, exc = S.T
    e = F - Strue
    vol = np.zeros(len(t), bool)
    for a, b, v in tr['segs']:
        if v: vol |= (t >= a) & (t < b)
    since = np.full(len(t), 1e9); pre = np.zeros(len(t), bool)
    for o in tr['onsets']:
        mm = t >= o + 3; since[mm] = np.minimum(since[mm], t[mm] - o - 3); pre |= (t >= o) & (t < o + 3)
    calm = ~vol & ~pre & (since > 5); vv = vol & ~pre & (since > 5); post = since < 5
    r = lambda mk: float(np.sqrt((e[mk] ** 2).mean()))
    a = lambda mk: float(np.abs(e[mk]).mean())
    zz = lambda mk: float(np.sqrt(((e[mk] / np.maximum(sd[mk], 1e-3)) ** 2).mean()))
    return dict(calm_mae=a(calm), calm_rmse=r(calm), calm_z=zz(calm), vol_mae=a(vv), vol_rmse=r(vv), vol_z=zz(vv),
                post_mae=a(post), post1_mae=a(since < 1), pre_mae=a(pre),
                pv_hit=float((pv[vol] > .5).mean()), pv_fa=float((pv[~vol] > .5).mean()))

def job(args):
    strat, seed, over, t1 = args
    S, St, tr = run(strat, seed, over, t1)
    return score(S, St, tr)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('strat'); ap.add_argument('seeds')
    ap.add_argument('--grid', nargs='*', default=[]); ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--t1', type=float, default=6000); ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    base = {k: ast.literal_eval(v) for k, v in (kv.split('=', 1) for kv in a.set)}
    keys = [g.split('=')[0] for g in a.grid]; vals = [[ast.literal_eval(x) for x in g.split('=')[1].split(',')] for g in a.grid]
    combos = [dict(zip(keys, c)) for c in itertools.product(*vals)] or [{}]
    seeds = [int(s) for s in a.seeds.split(',')]
    tasks = [(a.strat, s, base | c, a.t1) for c in combos for s in seeds]
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(job, tasks))
    for ci, c in enumerate(combos):
        rs = res[ci * len(seeds):(ci + 1) * len(seeds)]
        avg = {k: np.mean([x[k] for x in rs]) for k in rs[0]}
        print(json.dumps(c), ' '.join(f'{k}={v:.3f}' for k, v in avg.items()), flush=True)
