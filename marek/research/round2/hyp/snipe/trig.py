import pickle, json, sys, bisect, pathlib
import numpy as np
D = pathlib.Path(__file__).parent; ROOT = D.parents[4]
PFX = sys.argv[1]
for tape in sys.argv[2:]:
    X = pickle.load(open(D / f'{PFX}_{tape}.pkl', 'rb'))
    e = np.array(X['eq'])
    if tape != 'r2':
        tr = json.load(open(ROOT / f'data/synth/r2_v2_seed{tape[1:]}.truth.json'))
        ons, js = tr['onsets'], tr['jumps']
    else:
        import csv
        ons = [float(r['onset']) for r in csv.DictReader(open(ROOT / 'marek/research/round2/flow/shock_events.csv'))]; js = [0]*len(ons)
    for (t, d, ref, sg, F) in X['trig']:
        k = min(range(len(ons)), key=lambda i: abs(ons[i] + 3 - t))
        i = bisect.bisect_left(e[:, 0], t)
        Fs = [e[min(i + j, len(e) - 1), 4] for j in (0, 1, 2, 4, 10)]
        q = sum(f[1] for f in X['fills'] if t <= f[0] < t + 0.2 and f[3] == 'ss')
        print(tape, f"t={t:.2f} dt={t-ons[k]:.3f} jump={js[k]:6.1f} dir={d:2d} ref={ref} F-ref={(F-ref)*d:5.1f} Fs-ref=" +
              ' '.join(f"{(x-ref)*d:5.1f}" for x in Fs) + f" ssQ={q}")
