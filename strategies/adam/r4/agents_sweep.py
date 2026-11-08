"""Full round-4 tape sweep for strategies/adam/r4/d_agents.py with the calm / volatile / shock split of
marek/research/round4/strategy/eval4.py (shock window = onset -1 .. +20 s, regimes from flow/regimes_1s.csv).

uv run python strategies/adam/r4/agents_sweep.py --tag T --cfg '{"ag_r15":1}' --cfg '{...}' [--file F] [--workers 8]
Rows go to strategies/adam/r4/agents_sweep.jsonl. Prints deltas vs the first config ({} = r4_best) as
calm / vol / cv (= calm + vol, the shock-lottery-free number) / shock / median per-shock delta / total.
"""
import argparse, json, pathlib, statistics, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'marek/research/round4/strategy'))
from eval4 import split, load_ctx  # noqa: E402
from fastsim.pool import Job, run_many  # noqa: E402

OUT = pathlib.Path(__file__).with_name('agents_sweep.jsonl')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', default='strategies/adam/r4/d_agents.py')
    ap.add_argument('--tape', default='data/round4_public.csv')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--cfg', action='append', default=[])
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    cfgs = [json.loads(c) for c in a.cfg] or [{}]
    t0 = time.time()
    outs = run_many([Job(a.file, a.tape, c) for c in cfgs], a.workers)
    ctx = load_ctx(a.tape)
    rows = []
    for c, o in zip(cfgs, outs):
        r = split(o, *ctx)
        row = {'tag': a.tag, 'file': a.file, 'tape': a.tape, 'set': c, 'secs': round(time.time() - t0)}
        if 'err' in r:
            row['err'] = r['err']
            print('ERR', c, r['err'][-800:])
        else:
            row.update(pnl=round(r['pnl']), breach=round(r.get('breach_penalty', 0)), parts=r['parts'],
                       win=r['win'], win_cv=r['win_cv'], per_shock=r['per_shock'])
        rows.append(row)
        with open(OUT, 'a') as fh:
            fh.write(json.dumps(row) + '\n')
    b = rows[0]
    print(f"{'config':72} {'calm':>7} {'vol':>7} {'cv':>7} {'tcv':>5} {'shock':>7} {'shmed':>6} {'total':>8}")
    for r in rows:
        if 'err' in r:
            continue
        P, B = r['parts'], b['parts']
        dc, dv, ds = (P['calm'] - B['calm']) / 1e3, (P['volatile'] - B['volatile']) / 1e3, (P['shock'] - B['shock']) / 1e3
        dw = [x - y for x, y in zip(r['win_cv'], b['win_cv'])]
        sd = statistics.stdev(dw) if any(dw) else 0
        tcv = statistics.fmean(dw) / (sd / len(dw) ** 0.5) if sd else 0.0
        med = statistics.median(x - y for x, y in zip(r['per_shock'], b['per_shock'])) / 1e3
        print(f"{json.dumps(r['set'])[:72]:72} {dc:+7.1f} {dv:+7.1f} {dc + dv:+7.1f} {tcv:+5.1f} {ds:+7.1f} {med:+6.2f} "
              f"{r['pnl'] / 1e3:8.1f}")
    print('done', len(rows), round(time.time() - t0), 's')


if __name__ == '__main__':
    main()
