"""Method check inputs for a synthetic r3 tape (strategy/r3_synth.py, known truth): ladder membership by exact 2000 us
maker chains, truth regime segments and shock onsets. Usage: TAPE=data/synth/r3_v1_seed1.csv synth_prep.py"""
import os, json, numpy as np, pandas as pd
import prep
tape = os.environ['TAPE']; stem = tape.split('/')[-1].replace('.csv', '')
d = prep.build(tape)
tu = d['t_us']; mk = ~d['ioc']; side = d['sign']
idx = np.where(mk)[0]
pos = {}
for j in idx: pos.setdefault((int(tu[j]), side[j]), []).append(j)
mem = np.full(len(tu), -1)
for j in idx:
    nxt = pos.get((int(tu[j]) + 2000, side[j]))
    if nxt:
        mem[j] = 1
        for k in nxt: mem[k] = 1
print('ladder members (chains >= 2):', (mem >= 0).sum())
np.save(prep.common.CACHE / f'{stem}_ladder.npy', np.c_[mem, mem])
T = json.load(open(prep.common.ROOT / tape.replace('.csv', '.truth.json')))
pd.DataFrame(dict(start=[a for a, b, v in T['segs'] if v], end=[b for a, b, v in T['segs'] if v])).to_csv(prep.common.CACHE / f'{stem}_vol.csv', index=False)
pd.DataFrame(dict(onset=T['onsets'])).to_csv(prep.common.CACHE / f'{stem}_shocks.csv', index=False)
print(list(T.keys()))
