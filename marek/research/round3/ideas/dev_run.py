"""dev_run.py P JOBFILE: each line "TAG FILE TAPESET [k=v ...]"; TAPESET = r3 | v1 | v2 | r2 | acc (r3+v1) | screen
(r3 + v1 seed 1) | all | comma list of tapes. Runs eval3.py once per (config, tape), at most P simulator processes at a
time (develop angle). Lines starting with # are skipped."""
import subprocess, sys, concurrent.futures as cf, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[4]
V1 = [f'data/synth/r3_v1_seed{i}.csv' for i in range(1, 7)]
V2 = [f'data/synth/r3_v2_seed{i}.csv' for i in range(11, 15)]
SETS = {'r3': ['data/round3.csv'], 'r2': ['data/round2.csv'], 'v1': V1, 'v2': V2, 'acc': ['data/round3.csv'] + V1,
        'screen': ['data/round3.csv', V1[0]], 'all': ['data/round3.csv'] + V1 + V2 + ['data/round2.csv'],
        'fin': V2 + ['data/round2.csv']}
P = int(sys.argv[1])
jobs = []
for line in open(sys.argv[2]):
    w = line.split()
    if not w or w[0].startswith('#'):
        continue
    tag, f, ts, kv = w[0], w[1], w[2], w[3:]
    for tp in SETS.get(ts, ts.split(',')):
        jobs.append([sys.executable if False else 'uv', 'run', 'python', 'marek/research/round3/strategy/eval3.py', f,
                     '--tape', tp, '--tag', tag] + (['--set'] + kv if kv else []))
def go(c):
    r = subprocess.run(c, cwd=ROOT, capture_output=True, text=True)
    return (r.stdout.strip() or r.stderr[-500:])[:220]
with cf.ThreadPoolExecutor(P) as ex:
    for out in ex.map(go, jobs):
        print(out, flush=True)
print('DONE', flush=True)
