"""Evaluate a strategy on round2 + round-2 synth seeds in parallel, with walk-forward windows.
uv run python marek/research/round2/hyp/eval.py STRATEGY.py [--set key=val ...] [--tapes r2,s1,...] [--tag name] [--base TAG]
--set appends PARAMS.update({...}) to a temp copy of the strategy.
Each tape's pnl.csv is cut into 12 windows of 500 s (walk-forward folds). r2 'train' = 0-4500 s, 'hold' = 4500-6000 s
(shock 5398 lives in hold; r2_ex5398 = r2 minus the [5394,5402] s shock window). --base TAG compares window-by-window with a previous row of that tag (paired deltas).
Results appended to marek/research/round2/hyp/results.jsonl. pnl is already net of breach penalty.
"""
import argparse, json, subprocess, tempfile, time, pathlib, ast, math, concurrent.futures as cf
ROOT = pathlib.Path(__file__).resolve().parents[4]
RES = pathlib.Path(__file__).parent / 'results.jsonl'
TAPES = {'r2': 'data/round2.csv', **{f's{i}': f'data/synth/r2_v2_seed{i}.csv' for i in range(1, 8)}}
EDGES = [500 * i for i in range(13)]
def run(strat, tape, logdir):
    subprocess.run(['uv', 'run', 'exchange-simulator', '--strategy', strat, '--backtest', str(ROOT / TAPES[tape]),
                    '--log-dir', logdir, '--no-logs'], cwd=ROOT, capture_output=True, text=True)
    try:
        r = json.load(open(pathlib.Path(logdir) / 'result.json'))
    except Exception:
        return None
    try:
        rows = [l.split(',') for l in open(pathlib.Path(logdir) / 'pnl.csv').read().split('\n')[1:] if l]
        ts = [float(a) for a, b in rows]; ps = [float(b) for a, b in rows]
        marks, j = [], 0
        for e in EDGES:
            while j + 1 < len(ts) and ts[j + 1] <= e + 1e-6: j += 1
            marks.append(ps[j])
        marks[-1] = r['pnl']
        r['win'] = [round(marks[i + 1] - marks[i], 1) for i in range(12)]
        if tape == 'r2':  # PnL inside the big 5398 shock window [5394, 5402] s, to judge overfitting to that one event
            a = max(i for i, t in enumerate(ts) if t <= 5394.0); b = max(i for i, t in enumerate(ts) if t <= 5402.0)
            r['sh5398'] = round(ps[b] - ps[a], 1)
    except Exception:
        r['win'] = None
    return r
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('strategy'); ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--tapes', default='r2,s1,s2,s3,s4,s5,s6,s7'); ap.add_argument('--tag', default='')
    ap.add_argument('--base', default='')
    a = ap.parse_args()
    over = {}
    for kv in a.set:
        k, v = kv.split('=', 1); over[k] = ast.literal_eval(v)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix='hyp_'))
    src = pathlib.Path(a.strategy).read_text()
    if over:
        src += f'\n\nPARAMS.update({over!r})\n'
    strat = tmp / pathlib.Path(a.strategy).name; strat.write_text(src)
    tapes = a.tapes.split(','); t0 = time.time()
    with cf.ThreadPoolExecutor(len(tapes)) as ex:
        res = dict(zip(tapes, ex.map(lambda t: run(str(strat), t, str(tmp / t)), tapes)))
    pnl = {t: (r['pnl'] if r else None) for t, r in res.items()}
    win = {t: r.get('win') for t, r in res.items() if r}
    syn = [v for t, v in pnl.items() if t != 'r2' and v is not None]
    row = {'tag': a.tag, 'strategy': a.strategy, 'set': over, 'pnl': pnl,
           'synth_mean': sum(syn) / len(syn) if syn else None,
           'breach': {t: round(r['breach_penalty']) for t, r in res.items() if r and r['breach_penalty']},
           'win': win, 'secs': round(time.time() - t0)}
    if win.get('r2'):
        row['r2_train'] = round(sum(win['r2'][:9])); row['r2_hold'] = round(sum(win['r2'][9:]))
    if res.get('r2') and res['r2'].get('sh5398') is not None:
        row['r2_sh5398'] = round(res['r2']['sh5398']); row['r2_ex5398'] = round(res['r2']['pnl'] - res['r2']['sh5398'])
    if a.base:
        base = None
        for l in RES.read_text().splitlines():
            try:
                b = json.loads(l)
            except Exception:
                continue
            if b.get('tag') == a.base and b.get('win'): base = b
        if base:
            d, cmp = [], {}
            for t, w in win.items():
                bw = base['win'].get(t)
                if w and bw:
                    dd = [x - y for x, y in zip(w, bw)]; d += dd
                    cmp[t] = {'d': round(sum(dd)), 'up': sum(x > 0 for x in dd), 'dn': sum(x < 0 for x in dd)}
            if 'r2' in cmp:
                dd = [x - y for x, y in zip(win['r2'], base['win']['r2'])]
                cmp['r2']['train'] = round(sum(dd[:9])); cmp['r2']['hold'] = round(sum(dd[9:]))
                if row.get('r2_ex5398') is not None and base.get('r2_ex5398') is not None:
                    cmp['r2']['ex5398'] = row['r2_ex5398'] - base['r2_ex5398']
            n = len(d)
            if n > 1:
                m = sum(d) / n; sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
                row['vs_base'] = {'base': a.base, 'per_tape': cmp, 'win_mean': round(m), 'win_t': round(m / (sd / math.sqrt(n) + 1e-9), 2),
                                  'win_up': sum(x > 0 for x in d), 'win_dn': sum(x < 0 for x in d)}
    with open(RES, 'a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps({k: v for k, v in row.items() if k != 'win'}))
if __name__ == '__main__': main()
