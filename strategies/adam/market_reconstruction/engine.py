"""FIFO book and identical measurements for observed and generated market tapes."""
from array import array
from bisect import insort
from collections import deque
import numpy as np


class Book:
    def __init__(self):
        self.levels = [{}, {}]
        self.quantities = [{}, {}]
        self.prices = [[], []]
        self.live = {}

    def best(self, side):
        p = self.prices[side]
        return (p[-1] if side == 0 else p[0]) if p else 0

    def qty(self, side):
        return self.quantities[side].get(self.best(side), 0)

    def depth(self, side, ticks=5):
        best = self.best(side)
        return sum(q for p, q in self.quantities[side].items() if abs(p-best)<ticks)

    def cancel(self, oid):
        found = self.live.pop(oid, None)
        if found is None:
            return False
        s, p, entry = found
        self.levels[s][p].remove(entry)
        self.quantities[s][p] -= entry[1]
        if not self.levels[s][p]:
            del self.levels[s][p], self.quantities[s][p]
            self.prices[s].remove(p)
        return True

    def insert(self, oid, side, price, volume, ioc):
        opp = 1-side
        remaining, value, fills, worst = volume, 0, 0, 0
        initial_touch = self.best(opp)
        while remaining and self.prices[opp]:
            best = self.best(opp)
            if (side == 0 and price<best) or (side == 1 and price>best):
                break
            queue = self.levels[opp][best]
            while remaining and queue:
                maker = queue[0]
                q = min(remaining,maker[1])
                remaining -= q
                maker[1] -= q
                value += q*best
                fills += 1
                worst = best
                self.quantities[opp][best] -= q
                if not maker[1]:
                    queue.popleft()
                    del self.live[maker[0]]
            if not queue:
                del self.levels[opp][best], self.quantities[opp][best]
                self.prices[opp].remove(best)
        if remaining and not ioc:
            if price not in self.levels[side]:
                self.levels[side][price] = deque()
                self.quantities[side][price] = 0
                insort(self.prices[side],price)
            entry = [oid,remaining]
            self.levels[side][price].append(entry)
            self.quantities[side][price] += remaining
            self.live[oid] = (side,price,entry)
        return volume-remaining,fills,abs(worst-initial_touch) if fills else 0,value


class Recorder:
    def __init__(self,duration):
        self.duration = duration
        self.next_grid = 0.0
        for name in ('grid_t','mid','spread','touch_qty','depth_qty','bid_qty','ask_qty',
                     'event_t','g_t','g_size','ioc_t','ioc_size','ioc_filled','ioc_side',
                     'ioc_depth','ioc_fills','ioc_vwap','cancel_live'):
            setattr(self,name,array('d'))
        self.event_kind = array('b')

    def advance(self,t,book):
        while self.next_grid<=t and self.next_grid<self.duration:
            b,a = book.best(0),book.best(1)
            valid = b>0 and a>b
            self.grid_t.append(self.next_grid)
            self.mid.append((b+a)/2 if valid else np.nan)
            self.spread.append(a-b if valid else np.nan)
            self.touch_qty.append((book.qty(0)+book.qty(1))/2 if valid else np.nan)
            self.depth_qty.append((book.depth(0)+book.depth(1))/2 if valid else np.nan)
            self.bid_qty.append(book.qty(0))
            self.ask_qty.append(book.qty(1))
            self.next_grid = round(self.next_grid+.1,6)

    def insert(self,t,oid,side,price,volume,ioc,book):
        self.event_t.append(t)
        self.event_kind.append(2 if ioc else 1)
        result = book.insert(oid,side,price,volume,ioc)
        if ioc:
            filled,fills,depth,value = result
            self.ioc_t.append(t)
            self.ioc_size.append(volume)
            self.ioc_filled.append(filled)
            self.ioc_side.append(side)
            self.ioc_depth.append(depth)
            self.ioc_fills.append(fills)
            self.ioc_vwap.append(value/filled if filled else np.nan)
        else:
            self.g_t.append(t)
            self.g_size.append(volume)
        return result

    def cancel(self,t,oid,book):
        self.event_t.append(t)
        self.event_kind.append(0)
        self.cancel_live.append(book.cancel(oid))

    def data(self):
        return {key:np.asarray(v) for key,v in self.__dict__.items() if isinstance(v,array)}

    def save(self,path):
        np.savez_compressed(path,**self.data(),duration=self.duration)


def acf(x):
    x = np.asarray(x)
    ok = np.isfinite(x[:-1]) & np.isfinite(x[1:])
    return float(np.corrcoef(x[:-1][ok],x[1:][ok])[0,1]) if ok.sum()>2 and np.nanstd(x)>0 else np.nan


def summarize(rec,start,end):
    grid = np.asarray(rec['grid_t'])
    mask = (grid>=start)&(grid<end)
    mid,spread = rec['mid'][mask],rec['spread'][mask]
    duration = end-start
    et,kinds = rec['event_t'],rec['event_kind']
    em = (et>=start)&(et<end)
    gm = (rec['g_t']>=start)&(rec['g_t']<end)
    it = rec['ioc_t']
    im = (it>=start)&(it<end)
    sizes,filled = rec['ioc_size'][im],rec['ioc_filled'][im]
    traded = filled>0
    result = dict(events_per_s=float(em.sum()/duration),limit_per_s=float(gm.sum()/duration),
                  ioc_per_s=float(im.sum()/duration),cancels_per_s=float(np.count_nonzero(em&(kinds==0))/duration),
                  mean_limit_size=float(rec['g_size'][gm].mean()),mean_ioc_size=float(sizes.mean()),
                  ioc_size_p99=float(np.quantile(sizes,.99)),ioc_fill_probability=float(traded.mean()),
                  executed_size_mean=float(filled[traded].mean()),executed_size_p99=float(np.quantile(filled[traded],.99)),
                  traded_lots_per_s=float(filled.sum()/duration),mean_sweep_ticks=float(rec['ioc_depth'][im][traded].mean()),
                  spread_mean=float(np.nanmean(spread)),spread_median=float(np.nanmedian(spread)),
                  spread_one_tick_share=float((spread[np.isfinite(spread)]==1).mean()),
                  one_sided_share=float((~np.isfinite(mid)).mean()),
                  mean_touch_quantity=float(np.nanmean(rec['touch_qty'][mask])),
                  mean_depth_5ticks=float(np.nanmean(rec['depth_qty'][mask])),
                  returns_acf_100ms=acf(np.diff(mid)),returns_acf_1s=acf(np.diff(mid[::10])),
                  ioc_side_acf=acf(rec['ioc_side'][im][traded]))
    for step,label in ((1,'100ms'),(10,'1s'),(100,'10s')):
        result['sigma_'+label] = float(np.sqrt(np.nanmean(np.diff(mid[::step])**2)/(step*.1)))
    for width,label in ((.1,'100ms'),(1.,'1s'),(10.,'10s')):
        n = int(duration/width)
        c = np.histogram(et[em]-start,bins=np.arange(n+1)*width)[0]
        result['fano_'+label] = float(c.var(ddof=1)/c.mean())
    future_index = np.searchsorted(grid,it[im]+1,side='right')-1
    valid = traded & (it[im]+1<end) & (future_index>=0)&(future_index<len(grid))
    future = rec['mid'][np.clip(future_index,0,len(grid)-1)]
    maker_sign = np.where(rec['ioc_side'][im]==0,-1,1)
    markout = maker_sign*(future-rec['ioc_vwap'][im])
    result['maker_markout_1s'] = float(np.nanmean(markout[valid]))
    return result
