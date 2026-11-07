import numpy as np, pandas as pd
from prep import load
o=load()
w=o[(o.t_us>3000e6)&(o.t_us<3000.6e6)]
pd.set_option('display.width',200)
print(w[['t_us','sg','price','volume','ioc','filled','bid','ask']].to_string())
