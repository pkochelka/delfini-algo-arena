"""Print the GA leaderboard from ledger.jsonl (latest record per id+set wins). usage: python board.py [set] [n]"""
import json, pathlib, sys

GA = pathlib.Path(__file__).resolve().parent
st = sys.argv[1] if len(sys.argv) > 1 else 'train'
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
recs = {}
for line in open(GA / 'ledger.jsonl'):
    r = json.loads(line)
    if r.get('set', 'train') == st:
        recs[r['id']] = r
rows = sorted(recs.values(), key=lambda r: -r['fitness'])[:n]
for r in rows:
    pn = r.get('pnl') or {}
    err = ' ERR' if r.get('error') else ''
    print(f"{r['id']:<10} {r['fitness']:>10} min={r.get('min')!s:>8} parent={r.get('parent', ''):<8} "
          + ' '.join(f'{v / 1e3:6.0f}' if v is not None else '  None' for v in pn.values()) + f"{err}  {r.get('desc', '')[:70]}")
