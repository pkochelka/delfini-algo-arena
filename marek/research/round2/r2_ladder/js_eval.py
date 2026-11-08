"""Score the post-jump sweep's direction calls against the true jumps.

usage: js_eval.py <run name> <tape stem> [...]   (run with debug=1; reads DBG lines from stdout.txt)
"""
import csv, json, re, sys
JR2 = [11.5, 12, -5.5, -6, -12.5, -17.5, 52.5, -20, -3.5, -33.5, 35.5, -55.5, 29, 31.5, -54.5, -24, -88.5, -12,
       -51, 22.5, 10.5, 47.5, 5, 1.5, 114.5, 100, -3.5]


def truth(stem):
    if stem == 'round2':
        on = [float(r['onset']) for r in csv.DictReader(open('marek/research/round2/flow/shock_events.csv'))]
        return list(zip(on, JR2))
    j = json.load(open(f'data/synth/{stem}.truth.json'))
    return list(zip(j['onsets'], j['jumps']))


args = sys.argv[1:]
for name, stem in zip(args[::2], args[1::2]):
    tr = truth(stem)
    txt = open(f'marek/logs/r2/{stem}/{name}/stdout.txt').read()
    shocks = [float(x) for x in re.findall(r'DBG shock \d+ t=([\d.]+)', txt)]
    jumps = [(float(t), int(d)) for d, t in re.findall(r'DBG jump dir=(-?\d) t=([\d.]+)', txt)]
    ok = bad = miss = early = 0
    big_bad = []
    lags = []
    dl = []
    for o, J in tr:
        det = [s for s in shocks if o - 0.5 < s < o + 1.5]
        if det:
            lags.append(det[0] - o)
        js = [(t, d) for t, d in jumps if o < t < o + 5]
        if not js:
            miss += 1
            continue
        t, d = js[0]
        lags.append(None)
        dl.append(t - o - 3.0)
        if t < o + 3.0:
            early += 1
        if d == (1 if J > 0 else -1):
            ok += 1
        else:
            bad += 1
            big_bad.append(round(J, 1))
    fp = len([s for s in shocks if not any(o - 0.5 < s < o + 1.5 for o, _ in tr)])
    lags = [x for x in lags if x is not None]
    lg = sorted(lags)
    dd = sorted(x for x in dl if x >= 0)
    dmed = dd[len(dd) // 2] * 1000 if dd else float('nan')
    print(f'{stem:<14} {name:<14} ok={ok} wrong={bad} {big_bad} none={miss} early={early} false_det={fp} '
          f'decision delay after jump med={dmed:.0f} ms')
