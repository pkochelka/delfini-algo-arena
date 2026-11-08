"""Many (file, --set) configs x tapes in one worker pool; rows appended to results.jsonl exactly like eval3.
uv run python marek/research/round3/strategy/sweep.py FILE --tag T --tapes a.csv,b.csv --cfg '{"theta":0.2}' --cfg '{...}'
--cfgfile F: one JSON dict per line. An empty dict {} = the file's own PARAMS."""
import argparse, json, sys, time, pathlib
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[3]))
from eval3 import split, load_ctx, RES
from fastsim.pool import Job, run_many
def main():
    global a, t0
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+'); ap.add_argument('--tag', required=True)
    ap.add_argument('--tapes', default='data/round3.csv,data/synth/r3_v1_seed1.csv,data/synth/r3_v1_seed2.csv,data/synth/r3_v1_seed3.csv,data/synth/r3_v2_seed11.csv,data/synth/r3_v2_seed12.csv')
    ap.add_argument('--cfg', action='append', default=[]); ap.add_argument('--cfgfile'); ap.add_argument('--workers', type=int, default=None)
    a = ap.parse_args()
    cfgs = [json.loads(c) for c in a.cfg]
    if a.cfgfile:
        cfgs += [json.loads(l) for l in open(a.cfgfile) if l.strip()]
    cfgs = cfgs or [{}]
    tapes = a.tapes.split(',')
    jobs = [Job(f, tp, c) for f in a.files for c in cfgs for tp in tapes]
    t0 = time.time()
    def emit(o):
        j = o['job']
        r = split(o, *load_ctx(j.tape))
        row = {'tag': a.tag, 'file': j.strategy, 'tape': j.tape, 'set': j.params, 'secs': round(time.time() - t0)}
        if 'err' in r:
            row['err'] = r['err']; print('ERR', j, r['err'][-500:], flush=True)
        else:
            row.update(pnl=round(r['pnl']), breach=round(r.get('breach_penalty', 0)), parts=r['parts'], win=r['win'], per_shock=r['per_shock'])
            print(j.tape.split('/')[-1], json.dumps(j.params), row['pnl'], flush=True)
        with open(RES, 'a') as fh:
            fh.write(json.dumps(row) + '\n')
    run_many(jobs, a.workers, on_result=emit)
    print('done', len(jobs), 'jobs', round(time.time() - t0), 's')

if __name__ == '__main__':
    main()
