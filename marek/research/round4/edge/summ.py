"""Summary of results.jsonl rows with deltas vs a base row. uv run python summ.py TAG [TAG ...] [--base TAG] [--tape round4]
Prints pnl, calm, vol, shock, breach, delta vs base (total/calm/vol/shock), and shock delta without the biggest |shock delta|."""
import json, sys, pathlib, argparse
ap = argparse.ArgumentParser(); ap.add_argument('tags', nargs='+'); ap.add_argument('--base', default='base4')
ap.add_argument('--tape', default='round4'); ap.add_argument('--strip', nargs='*', default=['"lad_hist":1,"lad_step":8,'])
ap.add_argument('--w', type=int, default=60); ap.add_argument('--baseset', default=None, help='JSON of the base row set (default: last row of the base tag)'); a = ap.parse_args()
rows = [json.loads(l) for l in open(pathlib.Path(__file__).parent / 'results.jsonl')]
rows = [r for r in rows if 'pnl' in r and a.tape in r['tape']]
base = [r for r in rows if r['tag'] == a.base and (a.baseset is None or r['set'] == json.loads(a.baseset))][-1]
def d(r):
    P = r['parts']; B = base['parts']
    ds = [x - y for x, y in zip(r['per_shock'], base['per_shock'])]
    big = max(ds, key=abs) if ds else 0; r['_bi'] = (ds.index(big), round(big/1e3,1)) if ds else None
    import statistics; med = statistics.median(ds) if ds else 0
    wd = [x - y for x, y in zip(r['win'], base['win'])]; mw = sum(wd) / 12; sw = (sum((x - mw) ** 2 for x in wd) / 11) ** 0.5
    tw = mw / (sw / 12 ** 0.5) if sw else 0
    if 'win_cv' in r and 'win_cv' in base:
        wd = [x - y for x, y in zip(r['win_cv'], base['win_cv'])]; mw = sum(wd) / 12; sw = (sum((x - mw) ** 2 for x in wd) / 11) ** 0.5
        tcv = mw / (sw / 12 ** 0.5) if sw else 0
    else:
        tcv = float('nan')
    wins = sum(x > y for x, y in zip(r['win'], base['win']))
    wcv = sum(x > y for x, y in zip(r.get('win_cv', []), base.get('win_cv', [])))
    return (f"{r['pnl']/1e3:8.1f} c{P['calm']/1e3:6.1f} v{P['volatile']/1e3:6.1f} s{P['shock']/1e3:6.1f} | "
            f"d {(r['pnl']-base['pnl'])/1e3:+6.1f} c{(P['calm']-B['calm'])/1e3:+6.1f} v{(P['volatile']-B['volatile'])/1e3:+6.1f} "
            f"s{(P['shock']-B['shock'])/1e3:+6.1f} s-big{(P['shock']-B['shock']-big)/1e3:+6.1f} smed{med/1e3:+5.2f} cv{(P['calm']+P['volatile']-B['calm']-B['volatile'])/1e3:+6.1f} w{wins:2d}/12 wcv{wcv:2d}/12 t{tw:+.1f} tcv{tcv:+.1f} big{r['_bi']}")
print('BASE', base['file'].split('/')[-1], d(base))
for r in rows:
    if r['tag'] in a.tags:
        st = json.dumps(r["set"]).replace(" ", "")
        for k in a.strip:
            st = st.replace(k, '')
        print(f"{r['tag']:6s} {st[:a.w]:{a.w}s} {d(r)}")
