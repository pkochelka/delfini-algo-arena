exec(open('an1.py').read().split('# fills')[0].replace("samp = np", "samp0 = np").replace("rg = np", "#").replace("for g in ('calm', 'vol', 'shock'):\n    m = rg", "if 0:\n    m = rg"))
import collections
fl = L['fill']
filled = collections.Counter()
for f in fl: filled[f[1]] += f[4]
# vol IOC sends: fill ratio
sv = [s for s in L['send'] if s[5] and reg(s[0]) == 'vol']
sq = sum(s[4] for s in sv); fq = sum(filled[s[1]] for s in sv)
print('vol IOC sends', len(sv), 'qty', sq, 'filled', fq, 'ratio %.2f' % (fq / sq))
# edge of sent vs filled: S(t+5) - lim
# take opportunities (log entries) in vol
tk = [x for x in L['take'] if reg(x[0]) == 'vol']
print('vol take log entries', len(tk))
# dedupe: episodes = same side, same touch price, gap < 0.05 s
eps = []
last = {}
for x in tk:
    t, sd, px, lim, q, room, F, fsd, turb, pos, pv = x
    k = sd
    e = last.get(k)
    if e is not None and e['px'] == px and t - e['t1'] < 0.05:
        e['t1'] = t; e['q'] = max(e['q'], q); e['room0'] = e['room0']; e['minroom'] = min(e['minroom'], room); e['n'] += 1
    else:
        e = dict(t0=t, t1=t, sd=sd, px=px, lim=lim, q=q, room0=room, minroom=room, F=F, pos=pos, n=1)
        eps.append(e); last[k] = e
print('episodes', len(eps))
import numpy as np
q = np.array([e['q'] for e in eps]); r0 = np.array([e['room0'] for e in eps]); sd = np.array([e['sd'] for e in eps])
t0 = np.array([e['t0'] for e in eps]); lim = np.array([e['lim'] for e in eps]); px = np.array([e['px'] for e in eps])
Fv = np.array([e['F'] for e in eps])
pinned = r0 <= 0
print('pinned at first sight: %.2f of episodes; qty avail total %d, qty in pinned eps %d' % (pinned.mean(), q.sum(), q[pinned].sum()))
miss = np.maximum(q - np.maximum(r0, 0), 0)
for h in (0.2, 1, 5, 15, 60):
    edge_touch = sd * (S(t0 + h) - px)  # edge on the touch level
    print('h', h, 'edge/lot at touch: all %.2f  pinned %.2f  room %.2f' % (np.average(edge_touch, weights=q), np.average(edge_touch[pinned], weights=q[pinned]), np.average(edge_touch[~pinned], weights=q[~pinned])))
print('missed qty (q - room) %d; F-edge of touch mean %.2f' % (miss.sum(), np.average(sd * (Fv - px), weights=q)))
# distribution of avail qty per episode
print('avail qty quantiles', np.percentile(q, [10, 25, 50, 75, 90, 99]))
print('room at first sight quantiles', np.percentile(r0, [10, 25, 50, 75, 90]))
# pinned eps by side relative to pos
