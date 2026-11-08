import sys, json, numpy as np
sys.path.insert(0, 'marek/research/round2/hyp/filter')
import synth_s
seed=int(sys.argv[1])
p=json.load(open(f'data/synth/r2_v2_seed{seed}.params.json')); p.pop('seed')
rows,truth=synth_s.generate(p,seed)
# verify against csv
ref=open(f'data/synth/r2_v2_seed{seed}.csv').read().split('\n')[1:]
ref=[r for r in ref if r]
print('rows',len(rows),len(ref),'match', rows==ref)
SP=np.array(truth['S']); np.save(f'marek/research/round2/hyp/filter/S_seed{seed}.npy',SP)
