"""Run r2_ladder variants on a tape in parallel and split PnL by calm / volatile / shock period.

usage: uv run python marek/research/round2/r2_ladder/run.py <tape> <name>='{"k": v}' [...]
Writes marek/logs/r2/<tape>/<name>/ and appends to marek/research/round2/r2_ladder/results.jsonl.
"""
import csv, json, subprocess, sys, pathlib
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[4]
SRC = ROOT / __import__('os').environ.get('STRAT', 'strategies/marek/round2/r2_ladder.py')
R = ROOT / 'marek/research/round2/flow/'
OUT = pathlib.Path(__file__).parent


def labeler(tape):
    tr = (ROOT / tape).with_suffix('.truth.json')
    if tr.exists():
        j = json.loads(tr.read_text())
        shocks, segs = j['onsets'], j['segs']
    else:
        shocks = [float(r['onset']) for r in csv.DictReader(open(R / 'shock_events.csv'))]
        segs = [(float(r['start']), float(r['end']), r['high'] == 'True') for r in csv.DictReader(open(R / 'taker_regimes.csv'))]

    def lab(t):
        for s in shocks:
            if s - 1 <= t < s + 20:
                return 'shock'
        for a, b, h in segs:
            if a <= t < b:
                return 'vol' if h else 'calm'
        return 'calm'
    return lab


def run(tape, name, over):
    d = ROOT / 'marek/logs/r2' / pathlib.Path(tape).stem / name
    d.mkdir(parents=True, exist_ok=True)
    src = SRC.read_text()
    params = {}
    for k, v in over.items():
        params[k] = v
    # inject overrides after PARAMS definition
    src = src.replace("\nBUY, SELL = Side.BUY, Side.SELL", "\nPARAMS.update(" + repr(params) + ")\nBUY, SELL = Side.BUY, Side.SELL", 1)
    strat = d / 'strategy.py'
    strat.write_text(src)
    out = subprocess.run(['uv', 'run', 'exchange-simulator', '--strategy', str(strat), '--backtest', str(ROOT / tape),
                          '--no-logs', '--log-dir', str(d)], capture_output=True, text=True, cwd=ROOT)
    (d / 'stdout.txt').write_text(out.stdout + out.stderr)
    res = json.loads((d / 'result.json').read_text())
    split = {}
    if 'round2' in tape or 'r2_' in tape:
        lab = labeler(tape)
        prev = 0.0
        for r in list(csv.DictReader(open(d / 'pnl.csv')))[1:]:
            t, pn = float(r['elapsed_s']), float(r['pnl'])
            l = lab(t - 0.5)
            split[l] = split.get(l, 0) + pn - prev
            prev = pn
    rec = {'tape': tape, 'name': name, 'params': over, 'pnl': res['pnl'], 'fills': res.get('fills'),
           'split': {k: round(v) for k, v in split.items()},
           'inv_q50': res.get('abs_inventory_q50'), 'breach': res.get('breach_penalty')}
    with open(OUT / 'results.jsonl', 'a') as f:
        f.write(json.dumps(rec) + '\n')
    return rec


if __name__ == '__main__':
    # tape may be a comma-separated list; with several tapes a per-config table is printed
    tapes = sys.argv[1].split(',')
    jobs = []
    for a in sys.argv[2:]:
        n, j = a.split('=', 1)
        jobs.append((n, json.loads(j)))
    allj = [(t, n, j) for n, j in jobs for t in tapes]
    with ThreadPoolExecutor(max_workers=int(__import__('os').environ.get('J', '9'))) as ex:
        recs = list(ex.map(lambda x: run(*x), allj))
    if len(tapes) == 1:
        for rec in recs:
            print(f"{rec['name']:<28} pnl={rec['pnl']:>11.0f} {rec['split']} invq50={rec['inv_q50']} fills={rec['fills']}")
    else:
        short = [pathlib.Path(t).stem.replace('r2_v1_', 'v1').replace('r2_v2_', 'v2') for t in tapes]
        print(f"{'config':<22}" + ''.join(f'{s:>10}' for s in short) + f"{'mean':>10}{'min':>10}")
        for n, _ in jobs:
            v = [r['pnl'] for r in recs if r['name'] == n]
            print(f'{n:<22}' + ''.join(f'{x:>10.0f}' for x in v) + f'{sum(v) / len(v):>10.0f}{min(v):>10.0f}')
