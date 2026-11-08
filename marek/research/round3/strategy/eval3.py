"""Run strategies on a round-3 tape and split PnL by calm / volatile / shock window.
uv run python marek/research/round3/strategy/eval3.py FILE [FILE ...] [--tape data/round3.csv] [--set k=v ...] [--tag T]
Shock window = onset-1 s .. onset+20 s (onsets: flow/shock_events.csv). Regimes: flow/regimes_1s.csv (1 s Viterbi).
12 x 500 s windows as walk-forward folds. Rows appended to strategy/results.jsonl. pnl is net of breach penalty.
"""
import argparse, ast, bisect, json, pathlib, subprocess, tempfile, time, concurrent.futures as cf, csv
ROOT = pathlib.Path(__file__).resolve().parents[4]
HERE = pathlib.Path(__file__).parent
RES = HERE / 'results.jsonl'
R3 = HERE.parent

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
    on = [float(r['onset']) for r in csv.DictReader(open(R3 / 'flow/shock_events.csv'))]
    reg = [(float(r['start']), float(r['end']), r['regime']) for r in csv.DictReader(open(R3 / 'flow/regimes_1s.csv'))]
    return on, reg

def label(t, on, reg):
    i = bisect.bisect_right(on, t + 1.0) - 1
    if i >= 0 and on[i] - 1.0 <= t < on[i] + 20.0:
        return 'shock'
    for a, b, g in reg:
        if a <= t < b:
            return g
    return 'calm'

def run(strat, tape, logdir, on, reg):
    p = subprocess.run(['uv', 'run', 'exchange-simulator', '--strategy', strat, '--backtest', str(ROOT / tape),
                        '--log-dir', logdir, '--no-logs'], cwd=ROOT, capture_output=True, text=True)
    try:
        r = json.load(open(pathlib.Path(logdir) / 'result.json'))
    except Exception:
        return {'err': p.stderr[-2000:]}
    rows = [l.split(',') for l in open(pathlib.Path(logdir) / 'pnl.csv').read().split('\n')[1:] if l]
    ts = [float(a) for a, b in rows]; ps = [float(b) for a, b in rows]
    parts = {'calm': 0.0, 'volatile': 0.0, 'shock': 0.0}
    per_shock = [0.0] * len(on)
    for k in range(1, len(ts)):
        g = label(ts[k - 1], on, reg); d = ps[k] - ps[k - 1]
        parts[g] += d
        if g == 'shock':
            i = bisect.bisect_right(on, ts[k - 1] + 1.0) - 1; per_shock[i] += d
    win = []
    for w in range(12):
        a = bisect.bisect_right(ts, 500 * w) - 1; b = bisect.bisect_right(ts, 500 * (w + 1)) - 1
        win.append(round(ps[max(b, 0)] - ps[max(a, 0)]))
    r['parts'] = {k: round(v) for k, v in parts.items()}
    r['per_shock'] = [round(x) for x in per_shock]
    r['win'] = win
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+'); ap.add_argument('--tape', default='data/round3.csv',
                                                         help='tape path, or comma list; "synth" = data/synth/r3_v1_seed1-6')
    ap.add_argument('--set', nargs='*', default=[]); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    tapes = [f'data/synth/r3_v1_seed{i}.csv' for i in range(1, 7)] if a.tape == 'synth' else a.tape.split(',')
    over = {}
    for kv in a.set:
        k, v = kv.split('=', 1); over[k] = ast.literal_eval(v)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix='r3_'))
    jobs = []
    for i, f in enumerate(a.files):
        src = pathlib.Path(f).read_text()
        if over:
            src += f'\n\nPARAMS.update({over!r})\n'
        s = tmp / f'{i}_{pathlib.Path(f).name}'; s.write_text(src)
        for j, tp in enumerate(tapes):
            jobs.append((f, str(s), str(tmp / f'log{i}_{j}'), tp))
    t0 = time.time()
    with cf.ThreadPoolExecutor(min(9, len(jobs))) as ex:
        out = list(ex.map(lambda j: (j[0], j[3], run(j[1], j[3], j[2], *load_ctx(j[3]))), jobs))
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
