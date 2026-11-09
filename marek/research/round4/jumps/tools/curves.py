"""Run strategies on round4 and save their PnL curves (elapsed_s, pnl) to cache/curve_<name>.npz.
uv run python marek/research/round4/jumps/tools/curves.py FILE [FILE ...] [--set k=v ...] [--suffix S]"""
import sys, ast, pathlib, argparse, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[5]; sys.path.insert(0, str(ROOT))
from fastsim.pool import Job, run_many
HERE = pathlib.Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('files', nargs='+'); ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--suffix', default=''); ap.add_argument('--tape', default='data/round4.csv')
    a = ap.parse_args()
    over = {k: ast.literal_eval(v) for k, v in (s.split('=', 1) for s in a.set)}
    outs = run_many([Job(f, a.tape, over) for f in a.files], 4)
    for f, o in zip(a.files, outs):
        if 'err' in o: print(o['err']); continue
        c = np.array(o['curve']); n = pathlib.Path(f).stem + a.suffix
        np.savez(HERE / 'cache' / f'curve_{n}.npz', t=c[:, 0], p=c[:, 1])
        print(n, len(c), c[-1], round(o['secs']))

if __name__ == '__main__':
    main()
