"""Independent checks of the research matching, exact tail sampler, and size likelihood."""
import tempfile
from pathlib import Path

import numpy as np
from scipy import special

import aggressive_size_fits as A
from analysis.replay import replay_csv

with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
    path = Path(folder)/'sizes.csv'
    path.write_text('Time,Operation,OrderId,Side,Volume,Price,Lifespan\n'
                    '0,Insert,b,B,10,100,G\n0,Insert,a,A,10,102,G\n'
                    '.01,Insert,t1,A,6,100,F\n.02,Cancel,b,,,,\n'
                    '.03,Insert,t2,A,6,99,F\n.04,Insert,b2,B,3,99,G\n'
                    '.05,Insert,t3,A,5,99,F\n.06,Insert,t4,B,20,102,F\n')
    streamed,crossed = A.read_ioc_sizes(path)
    replayed = replay_csv(path)
    fills = replayed.trades.groupby('taker_row').qty.sum()
    expected = fills.reindex(streamed.event_row).fillna(0).to_numpy(dtype=int)
    np.testing.assert_array_equal(streamed.executed_size,expected)
    assert streamed.executed_size.tolist() == [6,0,3,10] and crossed == 0

x = A.zipf_sample(2.8,10,20_000)
fit = A.zipf_fit(x,10)
assert abs(fit['alpha']-2.8)<.06
target = special.zeta(2.8,20)/special.zeta(2.8,10)
assert abs((x>=20).mean()-target)<.015
assert x.min()>=10 and fit['D']<.02
rng = np.random.default_rng(21)
x = np.maximum(1,np.floor(rng.lognormal(4.1,1.4,30_000)+.5)).astype(int)
ln = A.lognormal_fit(x,offset=.5)
assert abs(ln['mu']-4.1)<.03 and abs(ln['sigma']-1.4)<.03
print('PASS: streaming/replay parity including unfilled IOC, exact integer-tail sampling, MLE parameter recovery.')
