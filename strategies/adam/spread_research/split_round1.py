"""Streaming wrapper around Peter's actual Book and write_window splitter functions.

Same chronological boundaries and original priority snapshots as split_data.py;
streaming avoids keeping all 4.4M CSV rows as Python strings simultaneously.
"""
import csv
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').exists() and (p / 'simulator').is_dir())
SOURCE = REPO/'data'/'round1.csv'
OUT = HERE/'splits'


def split(source, out, count=6, warmup=60, end_fraction=.75):
    spec = importlib.util.spec_from_file_location('peter_split_data', REPO/'strategies'/'peter'/'split_data.py')
    peter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(peter)
    with source.open('rb') as file:
        first = file.readline()
        t0 = float(file.readline().decode().split(',')[0])
        file.seek(max(0,source.stat().st_size-4096))
        t1 = float(file.read().decode().strip().splitlines()[-1].split(',')[0])
    boundary = t0+(t1-t0)*end_fraction
    first_trade = t0+warmup
    step = (boundary-first_trade)/count
    definitions = [dict(name=f'wf{k+1:02d}', start_s=max(t0,first_trade+k*step-warmup),
                        eval_start_s=first_trade+k*step,end_s=first_trade+(k+1)*step)
                   for k in range(count)]
    definitions.append(dict(name='test',start_s=boundary,eval_start_s=boundary+warmup,end_s=t1+1e-6))
    out.mkdir(parents=True,exist_ok=True)
    book = peter.Book()
    handles = []
    activated = set()
    try:
        with source.open(newline='') as file:
            reader = csv.reader(file)
            assert next(reader) == peter.HEADER
            for row in reader:
                t = float(row[0])
                for k,d in enumerate(definitions):
                    if k not in activated and t >= d['start_s']:
                        snapshot = book.snapshot()
                        path = out/(d['name']+'.csv')
                        peter.write_window(path,[],[],d['start_s'],d['end_s'],snapshot)
                        handle = path.open('a',newline='')
                        d['seeded'] = len(snapshot)
                        d['rows'] = 0
                        d['trade_start_us'] = round(d['eval_start_s']*1e6)
                        d['file'] = str(path)
                        handles.append((k,handle,csv.writer(handle,lineterminator='\n')))
                        activated.add(k)
                    if k in activated and d['start_s'] <= t < d['end_s']:
                        writer = next(w for key,_,w in handles if key==k)
                        writer.writerow(row)
                        d['rows'] += 1
                book.apply(row)
    finally:
        for _,handle,_ in handles:
            handle.close()
    metadata = dict(source=str(source),splitter=str(REPO/'strategies/peter/split_data.py'),
                    mode='streaming use of original Book and write_window',warmup_s=warmup,
                    held_out_from_s=boundary,source_start_s=t0,source_end_s=t1,windows=definitions)
    (out/'folds.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2),flush=True)
    return metadata


if __name__=='__main__':
    split(SOURCE,OUT)
