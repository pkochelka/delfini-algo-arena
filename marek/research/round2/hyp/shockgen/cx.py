import json, sys, pathlib
ROOT = pathlib.Path('/Users/marek/Documents/delfini-algo-arena')
for k in sys.argv[1:]:
    tr = json.load(open(ROOT/f'data/synth/r2_v2_seed{k}.truth.json'))
    ev = []; side = {}; px = {}
    W = [(o+3.0, j) for o, j in zip(tr['onsets'], tr['jumps'])]
    import bisect
    starts = [w[0] for w in W]
    good = bad = 0
    res = {i: [] for i in range(len(W))}
    with open(ROOT/f'data/synth/r2_v2_seed{k}.csv') as fh:
        next(fh)
        for line in fh:
            a = line.split(',')
            t = float(a[0])
            if a[1] == 'Insert':
                side[a[2]] = a[3]
            i = bisect.bisect_right(starts, t) - 1
            if i >= 0 and t - starts[i] <= 0.03 and a[1] == 'Cancel':
                res[i].append(side.get(a[2]))
    for i, (T, j) in enumerate(W):
        stale = 'A' if j > 0 else 'B'
        c = res[i][:4]
        print(k, round(T,2), round(j,1), ''.join(x or '?' for x in c), 'stale', stale)
