import pickle, sys, numpy as np
for tape in sys.argv[1:]:
    res = pickle.load(open(f'scan_{tape}.pkl', 'rb'))
    for c in (3, 4):
      for x in (5, 10):
        ev = []
        last = {'A': -1, 'B': -1}
        for te in sorted(res):
            r = res[te]
            for sd in 'AB':
                if (x, sd) not in r: continue
                L, V, Lb, Vb, n3, n4 = r[(x, sd)]
                if Lb <= 0: continue
                e = Vb / Lb
                net = min(Lb, 900) * (e - c)
                if te - last[sd] < 0.1:  # dedupe: same episode
                    if net > ev[-1][2]: ev[-1] = (te, sd, net, e, Lb)
                    continue
                last[sd] = te
                ev.append((te, sd, net, e, Lb))
        nets = np.array([z[2] for z in ev]); es = np.array([z[3] for z in ev])
        s = f"{tape} c={c} +{x}ms episodes={len(ev)}"
        for th in (5, 6, 8):
            m = es >= th
            s += f" | e>={th}: n={m.sum()} net={nets[m].sum():.0f} lots={sum(min(z[4],900) for z,mm in zip(ev,m) if mm)}"
        print(s)
