"""Round-4 copy of round3/strategy/eval3.py: round-4 tapes use round4/flow/{shock_events,regimes_1s}.csv,
round-3 tapes the round-3 files. Results go to round4/strategy/results.jsonl.
Run strategies on a round-3/4 tape and split PnL by calm / volatile / shock window.
uv run python marek/research/round3/strategy/eval4.py FILE [FILE ...] [--tape data/round3.csv] [--set k=v ...] [--tag T]
Shock window = onset-1 s .. onset+20 s (onsets: flow/shock_events.csv). Regimes: flow/regimes_1s.csv (1 s Viterbi).
12 x 500 s windows as walk-forward folds. Rows appended to strategy/results.jsonl. pnl is net of breach penalty.
Runs on fastsim (parquet tape cache, in-process worker pool; byte-identical to exchange-simulator, see fastsim/parity.py).
--jit compiles the strategies' JIT_KERNELS with numba; --workers sets the pool size (default cores - 1).
"""
import argparse, ast, bisect, json, pathlib, sys, time, csv
ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastsim.pool import Job, run_many
HERE = pathlib.Path(__file__).parent
RES = HERE / 'results.jsonl'
R4 = HERE.parent
R3 = R4.parent / 'round3'

def load_ctx(tape):
    tj = (ROOT / tape).with_suffix('.truth.json')
    if tj.exists():  # synthetic tape: exact shock onsets and regime segments
        tr = json.load(open(tj))
        return sorted(tr['onsets']), [(a, b, 'volatile' if v else 'calm') for a, b, v in tr['segs']]
    if 'round2' in tape:  # round-2 context, for like-for-like comparison
        R2 = R3.parent / 'round2'
        on = [float(r['onset']) for r in csv.DictReader(open(R2 / 'flow/shock_events.csv'))]
        reg = [(float(r['start']), float(r['end']), 'volatile' if r['high'] == 'True' else 'calm')
               for r in csv.DictReader(open(R2 / 'flow/taker_regimes.csv'))]
        return on, reg
    R = R4 if 'round4' in tape else R3
    on = [float(r['onset']) for r in csv.DictReader(open(R / 'flow/shock_events.csv'))]
    reg = [(float(r['start']), float(r['end']), r['regime']) for r in csv.DictReader(open(R / 'flow/regimes_1s.csv'))]
    return on, reg

def label(t, on, reg):
    i = bisect.bisect_right(on, t + 1.0) - 1
    if i >= 0 and on[i] - 1.0 <= t < on[i] + 20.0:
        return 'shock'
    for a, b, g in reg:
        if a <= t < b:
            return g
    return 'calm'

def split(out, on, reg):
    """Per-regime / per-shock / per-window PnL from one fastsim.pool output (result.json + pnl curve)."""
    if 'err' in out:
        return {'err': out['err']}
    r = dict(out['result'])
    ts = [a for a, b in out['curve']]; ps = [b for a, b in out['curve']]
    parts = {'calm': 0.0, 'volatile': 0.0, 'shock': 0.0}
    per_shock = [0.0] * len(on)
    wcv = [0.0] * 12
    for k in range(1, len(ts)):
        g = label(ts[k - 1], on, reg); d = ps[k] - ps[k - 1]
        parts[g] += d
        if g != 'shock':
            wcv[min(11, int(ts[k - 1] // 500))] += d
        if g == 'shock':
            i = bisect.bisect_right(on, ts[k - 1] + 1.0) - 1; per_shock[i] += d
    win = []
    for w in range(12):
        a = bisect.bisect_right(ts, 500 * w) - 1; b = bisect.bisect_right(ts, 500 * (w + 1)) - 1
        win.append(round(ps[max(b, 0)] - ps[max(a, 0)]))
    r['parts'] = {k: round(v) for k, v in parts.items()}
    r['per_shock'] = [round(x) for x in per_shock]
    r['win'] = win
    r['win_cv'] = [round(x) for x in wcv]
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+'); ap.add_argument('--tape', default='data/round4.csv',
                                                         help='tape path, or comma list; "synth" = data/synth/r3_v1_seed1-6')
    ap.add_argument('--set', nargs='*', default=[]); ap.add_argument('--tag', default='')
    ap.add_argument('--jit', action='store_true'); ap.add_argument('--workers', type=int, default=None)
    a = ap.parse_args()
    tapes = [f'data/synth/r3_v1_seed{i}.csv' for i in range(1, 7)] if a.tape == 'synth' else a.tape.split(',')
    over = {}
    for kv in a.set:
        k, v = kv.split('=', 1); over[k] = ast.literal_eval(v)
    jobs = [Job(f, tp, over, a.jit) for f in a.files for tp in tapes]
    t0 = time.time()
    outs = run_many(jobs, a.workers)
    out = [(j.strategy, j.tape, split(o, *load_ctx(j.tape))) for j, o in zip(jobs, outs)]
    for f, tp, r in out:
        row = {'tag': a.tag, 'file': f, 'tape': tp, 'set': over, 'secs': round(time.time() - t0)}
        if 'err' in r:
            row['err'] = r['err']
        else:
            row.update(pnl=round(r['pnl']), breach=round(r.get('breach_penalty', 0)), parts=r['parts'],
                       win=r['win'], per_shock=r['per_shock'])
        with open(RES, 'a') as fh:
            fh.write(json.dumps(row) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k not in ('per_shock',)}))

if __name__ == '__main__':
    main()
