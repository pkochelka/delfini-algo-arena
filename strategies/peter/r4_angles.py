"""Extra angles on round tapes: generator components (shocks, ladder, impact law, transient decay), momentum mechanism,
propagator / sign memory, realized spread, book resilience, Hurst / tails, trend episodes, momentum tradeability.
Usage: python r4_angles.py data/round3_public.csv data/round4_public.csv ..."""
import sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis import load_tape

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
RES = {}


def prep(csv):
    t = load_tape(csv, verbose=False); ev, o, tr = t.events, t.orders, t.trades
    Tus = ev.t_us.to_numpy(); T = Tus / 1e6
    bid = ev.bid.to_numpy().astype(float); ask = ev.ask.to_numpy().astype(float)
    two = (bid > 0) & (ask > 0)
    bid = np.where(bid > 0, bid, np.nan); ask = np.where(ask > 0, ask, np.nan)
    MID = pd.Series(np.where(two, (bid + ask) / 2, np.nan)).ffill().bfill().to_numpy()
    BID = pd.Series(bid).ffill().bfill().to_numpy(); ASK = pd.Series(ask).ffill().bfill().to_numpy()
    SPR = pd.Series(np.where(two, ask - bid, np.nan)).ffill().bfill().to_numpy()
    dur = T[-1]
    def at(x, arr=MID):  # value just after the last event at or before time x (s)
        return arr[np.clip(np.searchsorted(Tus, np.asarray(x) * 1e6, 'right') - 1, 0, len(arr) - 1)]
    g = np.arange(0, dur, 0.1); m = at(g)
    r1 = m[10::10] - m[:-10:10]
    rv10 = pd.Series(np.abs(np.diff(m))).rolling(100).sum().to_numpy()[9::10][:len(r1)]
    turb = np.nan_to_num(rv10) > 15
    f = o[o.ioc & (o.filled > 0)]
    return dict(ev=ev, o=o, tr=tr, Tus=Tus, T=T, MID=MID, BID=BID, ASK=ASK, SPR=SPR, at=at, m=m, r1=r1, turb=turb,
                dur=dur, f=f, ft=f.t_us.to_numpy() / 1e6, fs=np.where(f.side == 'B', 1.0, -1.0),
                fq=f.filled.to_numpy().astype(float), frow=f.row.to_numpy())


def tflag(D, x):  # turbulent flag of the second containing time x
    s = np.clip(np.asarray(x).astype(int), 0, len(D['turb']) - 1); return D['turb'][s]


# ---------------------------------------------------------------- A. shock schedule
def shocks(D, name):
    o = D['o']; gm = o[~o.ioc]; nb = int(D['dur'] * 10)
    c = np.bincount((gm.t_us.to_numpy() // 100_000).astype(int), minlength=nb)[:nb].astype(float)
    rate = pd.Series(c).rolling(10, center=True, min_periods=1).mean().to_numpy() * 10
    low = rate < 110
    st = np.flatnonzero(low & ~np.r_[False, low[:-1]]); en = np.flatnonzero(low & ~np.r_[low[1:], False])
    keep = (en - st + 1) >= 15
    st, en = st[keep] / 10, (en[keep] + 1) / 10
    at = D['at']
    J = at(st + 4.0) - at(st - 0.1)
    pre = at(st + 2.8) - at(st - 0.1)
    rows = dict(n=len(st), dur_med=np.median(en - st), dur_sd=np.std(en - st), first=st[0] if len(st) else np.nan,
                gap_mean=np.mean(np.diff(st)), gap_min=np.min(np.diff(st)), gap_max=np.max(np.diff(st)),
                drought_rate=np.mean([c[int(a*10):int(b*10)].mean() * 10 for a, b in zip(st, en)]),
                J_sd=J.std(), J_absmean=np.abs(J).mean(), J_small=(np.abs(J) < 3).mean(), pre_move_sd=pre.std())
    # time of the jump inside the drought: first 100 ms bin after onset where |mid - mid(onset)| > 3
    tj = []
    for a in st:
        gg = np.arange(a, a + 6, 0.01); mm = at(gg); k = np.flatnonzero(np.abs(mm - mm[0]) > max(3, 0))
        tj.append(gg[k[0]] - a if len(k) else np.nan)
    rows['jump_at_med'] = np.nanmedian(tj); rows['jump_at_iqr'] = np.nanpercentile(tj, 75) - np.nanpercentile(tj, 25)
    big = np.abs(J) >= 5
    for H in (1, 5, 30, 60):
        rows[f'cont{H}s'] = np.mean(np.sign(J[big]) * (at(st[big] + 4 + H) - at(st[big] + 4)))
    # are the regime-split jumps (|dmid|>=5 per 100 ms) outside droughts? share of jumps not near a drought
    RES.setdefault('shock', {})[name] = rows
    return st, en


# ---------------------------------------------------------------- B. ladder maker (exact 2.000 ms chains)
def ladder(D, name):
    o = D['o']; gm = o[~o.ioc].sort_values('t_us')
    t = gm.t_us.to_numpy(); sd = (gm.side == 'B').to_numpy(); px = gm.price.to_numpy()
    lo = np.searchsorted(t, t - 2000, 'left'); hi = np.searchsorted(t, t - 2000, 'right')
    ok = hi > lo; j = np.where(ok, lo, 0)
    link = ok & (sd[j] == sd) & (px == np.where(sd, px[j] - 1, px[j] + 1))
    RES.setdefault('ladder', {})[name] = dict(links_2ms=int(link.sum()), per_s=link.sum() / D['dur'],
                                              link_share_turb=tflag(D, t[link] / 1e6).mean() if link.any() else np.nan)


# ---------------------------------------------------------------- C. impact law + transient decay (price-only)
def impact(D, name):
    at, ft, fs, fq = D['at'], D['ft'], D['fs'], D['fq']
    nb = int(D['dur'] * 10); b = (ft * 10).astype(int)
    x = np.bincount(b, fs * np.sqrt(fq), minlength=nb)[:nb]     # signed sqrt volume per 100 ms
    n = np.bincount(b, minlength=nb)[:nb]
    m = D['m'][:nb + 1]; dm = np.diff(m)[:nb]
    tb = tflag(D, np.arange(nb) / 10)
    out = {}
    for lab, msk in (('calm', ~tb), ('turb', tb)):
        k = msk & (n > 0); beta = np.sum(x[k] * dm[k]) / np.sum(x[k] ** 2)
        out[f'c_sqrt_{lab}'] = beta
    # pure-maker moves: bins with no taker; momentum of those moves vs past 1 s move
    past = np.r_[np.zeros(10), m[10:nb] - m[:nb - 10]][:nb]
    k = (n == 0)
    out['nomk_beta_past1s'] = np.sum(past[k] * dm[k]) / np.sum(past[k] ** 2)
    k2 = (n > 0)
    out['tk_beta_past1s'] = np.sum(past[k2] * dm[k2]) / np.sum(past[k2] ** 2)
    out['share_absmove_nomk'] = np.abs(dm[k]).sum() / np.abs(dm).sum()
    # transient: regress future 10/30 s move on signed sqrt flow in the past 1 s (controls for the move itself)
    X = pd.Series(x).rolling(10).sum().fillna(0).to_numpy()
    for H in (10, 30):
        fut = np.r_[m[H * 10:nb] - m[:nb - H * 10], np.zeros(H * 10)][:nb]
        A = np.c_[X, past][:nb - H * 10]; y = fut[:nb - H * 10]
        bb = np.linalg.lstsq(A, y, rcond=None)[0]
        out[f'fut{H}_on_flow'] = bb[0]; out[f'fut{H}_on_past'] = bb[1]
    RES.setdefault('impact', {})[name] = out


# ---------------------------------------------------------------- D. positive-feedback takers
def feedback(D, name):
    ft, fs, at = D['ft'], D['fs'], D['at']
    past = at(ft - 0.01) - at(ft - 1.0)
    q = pd.cut(past, [-1e9, -10, -3, -1, -0.25, 0.25, 1, 3, 10, 1e9])
    tab = pd.Series(fs).groupby(q, observed=True).agg(['mean', 'size'])
    RES.setdefault('feedback', {})[name] = tab['mean'].round(3).to_dict()
    RES.setdefault('feedback_n', {})[name] = tab['size'].to_dict()


# ---------------------------------------------------------------- E. propagator / sign memory
def propagator(D, name):
    ft, fs, at, MID, frow = D['ft'], D['fs'], D['at'], D['MID'], D['frow']
    m0 = MID[np.maximum(frow - 1, 0)]
    tb = tflag(D, ft)
    out = {}
    for tau in (0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 60):
        r = fs * (at(ft + tau) - m0)
        out[f'R{tau:g}'] = r.mean(); out[f'R{tau:g}_calm'] = r[~tb].mean(); out[f'R{tau:g}_turb'] = r[tb].mean()
    RES.setdefault('prop', {})[name] = out
    s = fs - fs.mean(); v = s.var()
    lags = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    ac = np.array([np.mean(s[:-l] * s[l:]) / v for l in lags])
    pos = ac > 0
    gam = -np.polyfit(np.log(lags[pos][:7]), np.log(ac[pos][:7]), 1)[0] if pos.sum() >= 3 else np.nan
    sc = fs[~tb]; sc = sc - sc.mean(); acc = np.mean(sc[:-1] * sc[1:]) / sc.var()
    RES.setdefault('signacf', {})[name] = dict(zip([f'l{l}' for l in lags], ac.round(4))) | {'gamma': gam, 'calm_l1': acc}


# ---------------------------------------------------------------- F. realized spread of passive fills
def realized(D, name):
    tr, at, MID = D['tr'], D['at'], D['MID']
    t = tr.t_us.to_numpy() / 1e6; px = tr.price.to_numpy(); q = tr.qty.to_numpy().astype(float)
    sm = np.where(tr.aggressor.to_numpy() == 'A', 1.0, -1.0)   # maker bought when the aggressor sold
    dep = tr.depth.to_numpy(); m0 = MID[np.maximum(tr.taker_row.to_numpy() - 1, 0)]
    tb = tflag(D, t)
    rows = []
    for lab, msk in (('all', np.ones_like(tb)), ('calm', ~tb), ('turb', tb), ('touch', dep == 0), ('deep1+', dep >= 1)):
        r = {'set': lab, 'lots': q[msk].sum()}
        r['half_spread'] = np.sum(q[msk] * sm[msk] * (m0[msk] - px[msk])) / q[msk].sum()
        for H in (0.1, 1, 5, 30):
            r[f'RS{H:g}'] = np.sum(q[msk] * sm[msk] * (at(t[msk] + H) - px[msk])) / q[msk].sum()
        rows.append(r)
    RES.setdefault('realized', {})[name] = pd.DataFrame(rows).set_index('set')


# ---------------------------------------------------------------- G. resilience after sweeps
def resilience(D, name):
    ft, fs, frow, at, SPR, MID = D['ft'], D['fs'], D['frow'], D['at'], D['SPR'], D['MID']
    sp_after = SPR[frow]; sw = sp_after >= 2
    sp_before = SPR[np.maximum(frow - 1, 0)]; sw &= sp_before == 1
    out = {'sweeps_per_min': sw.sum() / D['dur'] * 60}
    for H in (0.01, 0.05, 0.1, 0.5, 1, 2):
        out[f'spr{H:g}'] = at(ft[sw] + H, SPR).mean()
    ma = MID[frow[sw]]
    for H in (0.1, 1, 5):
        out[f'cont{H:g}'] = np.mean(fs[sw] * (at(ft[sw] + H) - ma))
    RES.setdefault('resil', {})[name] = out


# ---------------------------------------------------------------- H. Hurst (DFA) + tails
def dfa(x, scales):
    y = np.cumsum(x - x.mean()); F = []
    for s in scales:
        n = len(y) // s; Y = y[:n * s].reshape(n, s); tt = np.arange(s)
        c = np.polyfit(tt, Y.T, 1); fit = np.outer(c[0], tt) + c[1][:, None]
        F.append(np.sqrt(np.mean((Y - fit) ** 2)))
    return np.polyfit(np.log(scales), np.log(F), 1)[0]


def hurst(D, name):
    r1, turb = D['r1'], D['turb']
    a = np.sort(np.abs(r1))[::-1]; k = int(0.01 * len(a)); hill = 1 / np.mean(np.log(a[:k] / a[k]))
    rc = r1[~turb]; ac = np.sort(np.abs(rc))[::-1]; kc = int(0.01 * len(ac)); hillc = 1 / np.mean(np.log(ac[:kc] / ac[kc]))
    RES.setdefault('hurst', {})[name] = dict(H_4_32s=dfa(r1, [4, 6, 8, 12, 16, 24, 32]), H_30_600s=dfa(r1, [30, 50, 80, 130, 200, 350, 600]),
                                             H_calm_4_32s=dfa(rc, [4, 6, 8, 12, 16, 24, 32]), hill_alpha=hill, hill_alpha_calm=hillc,
                                             kurt_calm=pd.Series(rc).kurt())


# ---------------------------------------------------------------- I. trend episodes
def episodes(D, name, st):
    """Turbulent episodes = maximal runs of turbulent seconds >= 10 s. Efficiency |net|/sqrt(sum r^2) (RW: ~0.8),
    and whether the first 5 s predicts the rest."""
    turb, r1, m = D['turb'], D['r1'], D['m']
    e = np.r_[0, turb.astype(int), 0]; s = np.flatnonzero(np.diff(e) == 1); t = np.flatnonzero(np.diff(e) == -1)
    keep = (t - s) >= 10; s, t = s[keep], t[keep]
    eff = np.array([abs(r1[a:b].sum()) / np.sqrt((r1[a:b] ** 2).sum()) for a, b in zip(s, t)])
    first = np.array([r1[a:a + 5].sum() for a in s]); rest = np.array([r1[a + 5:b].sum() for a, b in zip(s, t)])
    agree = np.mean(np.sign(first) == np.sign(rest))
    beta = np.sum(first * rest) / np.sum(first ** 2)
    # calm stretches too
    e2 = np.r_[0, (~turb).astype(int), 0]; s2 = np.flatnonzero(np.diff(e2) == 1); t2 = np.flatnonzero(np.diff(e2) == -1)
    k2 = (t2 - s2) >= 30
    eff2 = np.array([abs(r1[a:b].sum()) / np.sqrt((r1[a:b] ** 2).sum()) for a, b in zip(s2[k2], t2[k2])])
    RES.setdefault('episodes', {})[name] = dict(n_turb=len(s), len_med=np.median(t - s), eff_turb=eff.mean(), eff_calm=eff2.mean(),
                                                 first5_rest_agree=agree, rest_on_first5=beta)


# ---------------------------------------------------------------- J. momentum tradeability (IOC in, IOC out, 5 ms latency)
def tradeable(D, name):
    at, BID, ASK, dur = D['at'], D['BID'], D['ASK'], D['dur']
    g = np.arange(5, dur - 70, 0.05)
    out = []
    for L in (0.5, 1, 2):
        mv = at(g) - at(g - L)
        for X in (3, 5, 10, 20):
            for H in (1, 2, 5, 10, 30):
                idx = np.flatnonzero(np.abs(mv) >= X)
                sel = []; last = -1e9                         # non-overlapping trades
                for i in idx:
                    if g[i] >= last + H: sel.append(i); last = g[i]
                sel = np.array(sel, int)
                if len(sel) < 10: continue
                tt = g[sel]; sg = np.sign(mv[sel])
                ent = np.where(sg > 0, at(tt + 0.005, ASK), at(tt + 0.005, BID))
                ex = np.where(sg > 0, at(tt + H, BID), at(tt + H, ASK))
                exm = at(tt + H)
                pnl = sg * (ex - ent); pm = sg * (exm - ent)
                seg = (tt // (dur / 16)).astype(int)
                segp = pd.Series(pnl).groupby(seg).sum()
                out.append(dict(L=L, X=X, H=H, n=len(sel), pnl_lot=pnl.mean(), t=pnl.mean() / pnl.std() * np.sqrt(len(pnl)),
                                pnl_mid_exit=pm.mean(), total=pnl.sum(), splits_pos=(segp > 0).sum()))
    RES.setdefault('trade', {})[name] = pd.DataFrame(out)


def main():
  for csv in sys.argv[1:]:
      name = csv.split('/')[-1].replace('_public.csv', '').replace('.csv', '')
      D = prep(csv); print('loaded', name, flush=True)
      st, en = shocks(D, name); ladder(D, name); impact(D, name); feedback(D, name); propagator(D, name)
      realized(D, name); resilience(D, name); hurst(D, name); episodes(D, name, st); tradeable(D, name)

  for k in ('shock', 'ladder', 'impact', 'feedback', 'feedback_n', 'prop', 'signacf', 'resil', 'hurst', 'episodes'):
      print(f'\n===== {k}'); print(pd.DataFrame(RES[k]).round(4).to_string())
  for nm, df in RES['realized'].items():
      print(f'\n===== realized spread per lot (maker view, mid-based), {nm}'); print(df.round(3).to_string())
  for nm, df in RES['trade'].items():
      print(f'\n===== momentum IOC round trip, {nm}: best 12 by t, and the X=10 L=1 row')
      print(df.sort_values('t', ascending=False).head(12).round(3).to_string(index=False))
      print(df[(df.L == 1) & (df.X == 10)].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
