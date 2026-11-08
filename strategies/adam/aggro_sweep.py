"""Parameter sweep for strategies/adam/r3_aggro_spread.py on the round-3 tape (calm / volatile / shock split).

uv run python strategies/adam/aggro_sweep.py CONFIG_SET [--tape data/round3_public.csv] [--workers 7]
Config sets are defined in SETS below; rows are appended to strategies/adam/aggro_sweep.jsonl.
"""
import argparse, json, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'marek/research/round3/strategy'))
from fastsim.pool import Job, run_many  # noqa: E402
from eval3 import split, load_ctx  # noqa: E402

STRAT = 'strategies/adam/r3_aggro_spread.py'
OUT = pathlib.Path(__file__).with_name('aggro_sweep.jsonl')

SETS = {
    # round 1: each lever alone against the r3_hmm base
    's1': {
        'base': {},
        'tk_c05_sd05': {'take': 0.05, 'take_sd': 0.5},
        'tk_c00_sd05': {'take': 0.0, 'take_sd': 0.5},
        'tk_v15_sd05': {'take_t': 0.15, 'take_sd_t': 0.5},
        'tk_v00_sd05': {'take_t': 0.0, 'take_sd_t': 0.5},
        'trtk_m02': {'tr_take': -0.2, 'tr_take_sd': 0.5},
        'ws_e05': {'ws_on': 1, 'ws_edge': 0.5},
        'ws_e025': {'ws_on': 1, 'ws_edge': 0.25},
        'ws_e10': {'ws_on': 1, 'ws_edge': 1.0},
        'wst_m3_e1': {'wst_on': 1, 'wst_min': 3, 'wst_edge': 1.0},
        'wst_m2_e15': {'wst_on': 1, 'wst_min': 2, 'wst_edge': 1.5},
        'wst_red_m2_e05': {'wst_on': 1, 'wst_min': 2, 'wst_edge': 0.5, 'wst_keep': 0.0, 'wst_red': 1},
        'wtk_d03': {'wtk_on': 1, 'wtk_d': 0.3},
        'wtk_m3_d06': {'wtk_on': 1, 'wtk_min': 3, 'wtk_d': 0.6},
    },
    # round 2: refine the spread levers (s1: lower IOC thresholds all lose; ws_e025 +6.7k, wst_m2_e15 +0.5k)
    's2': {
        'base': {},
        'ws_e00': {'ws_on': 1, 'ws_edge': 0.0},
        'ws_e01': {'ws_on': 1, 'ws_edge': 0.1},
        'ws_e025_m3': {'ws_on': 1, 'ws_edge': 0.25, 'ws_min': 3},
        'ws_e025_sd05': {'ws_on': 1, 'ws_edge': 0.25, 'ws_sd': 0.5},
        'ws_e025_r00': {'ws_on': 1, 'ws_edge': 0.25, 'ws_edge_r': 0.0},
        'ws_e025_rm025': {'ws_on': 1, 'ws_edge': 0.25, 'ws_edge_r': -0.25},
        'wst_m2_e2': {'wst_on': 1, 'wst_min': 2, 'wst_edge': 2.0},
        'wst_m2_e25': {'wst_on': 1, 'wst_min': 2, 'wst_edge': 2.5},
        'wst_m2_e15_k1': {'wst_on': 1, 'wst_min': 2, 'wst_edge': 1.5, 'wst_keep': 1.0},
        'wst_m3_e15': {'wst_on': 1, 'wst_min': 3, 'wst_edge': 1.5},
        'wst_m2_e15_q30': {'wst_on': 1, 'wst_min': 2, 'wst_edge': 1.5, 'wst_q': 30},
        'tk_c10_sd075': {'take_sd': 0.75},
        'tk_v30_sd075': {'take_sd_t': 0.75},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('set')
    ap.add_argument('--tape', default='data/round3_public.csv')
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--only', nargs='*', default=None)
    a = ap.parse_args()
    cfgs = SETS[a.set]
    if a.only:
        cfgs = {k: v for k, v in cfgs.items() if k in a.only}
    jobs = [Job(STRAT, a.tape, over, False) for over in cfgs.values()]
    t0 = time.time()
    outs = run_many(jobs, a.workers)
    ctx = load_ctx(a.tape)
    rows = []
    for name, over, o in zip(cfgs, cfgs.values(), outs):
        r = split(o, *ctx)
        row = {'set': a.set, 'name': name, 'over': over, 'tape': a.tape, 'secs': round(time.time() - t0)}
        row.update({k: r[k] for k in ('pnl', 'breach_penalty', 'fills', 'traded_volume', 'parts', 'win', 'err')
                    if k in r})
        rows.append(row)
    with open(OUT, 'a') as fh:
        for row in rows:
            fh.write(json.dumps(row) + '\n')
    base = next((r.get('pnl') for r in rows if r['name'] == 'base'), None)
    print(f"{'name':<22}{'pnl':>10}{'d_base':>9}{'calm':>9}{'vol':>9}{'shock':>9}{'fills':>8}")
    for r in rows:
        if 'err' in r:
            print(r['name'], 'ERR', r['err'][:300])
            continue
        pt = r['parts']
        d = '' if base is None else f"{(r['pnl'] - base) / 1e3:+.1f}k"
        print(f"{r['name']:<22}{r['pnl'] / 1e3:>9.1f}k{d:>9}{pt['calm'] / 1e3:>8.1f}k{pt['volatile'] / 1e3:>8.1f}k"
              f"{pt['shock'] / 1e3:>8.1f}k{r.get('fills', 0):>8}")


if __name__ == '__main__':
    main()
