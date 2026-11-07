"""Fold runner that records our fills, plus the PnL decomposition used in research/round1/inventory.md.

run_rec(strategy, overrides, fold) -> dict(result fields, fills=(t_us, signed qty, price) arrays)
decompose(fills, t0_us, t1_us, final_pnl) -> spread capture / inventory PnL / drift split / pinning stats
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'strategies' / 'peter'))
sys.path.insert(0, str(ROOT))

_Q = None
TAPE = 'round1'  # which data/.cache/<TAPE>_quotes.npz mid_at uses (written by mid_cache.py <tape>)


def set_tape(name: str) -> None:
    global _Q, TAPE
    TAPE, _Q = name, None


def quotes():
    global _Q
    if _Q is None:
        z = np.load(ROOT / f'data/.cache/{TAPE}_quotes.npz')
        _Q = (z['t_us'], (z['bid'] + z['ask']) / 2, z['bid'], z['ask'])
    return _Q


def mid_at(t_us: np.ndarray) -> np.ndarray:
    t, mid, _, _ = quotes()
    i = np.searchsorted(t, t_us, side='right') - 1
    return mid[np.clip(i, 0, len(mid) - 1)]


def run_rec(strategy: str, overrides: dict, fold: dict) -> dict:
    from arena import load_agent
    from simulator.main import (MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS,
                                POSITION_BREACH_PENALTY_PER_SECOND)
    from simulator.models import OrderStatus, Side
    from simulator.simulator import ExchangeSimulator, SimulationConfig

    strat = load_agent(Path(strategy), overrides | {'trade_start_us': fold['trade_start_us']}, 'rec')
    rec: list[tuple[int, int, int]] = []
    orig = strat.on_private_order_update

    def hook(ev):
        o = strat.orders.get(ev.order_id)
        if o is not None and ev.status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED) and ev.filled_quantity:
            s = 1 if o[0] == Side.BUY else -1
            rec.append((ev.timestamp_us, s * ev.filled_quantity, ev.fill_price))
        orig(ev)

    strat.on_private_order_update = hook
    cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                           position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
                           order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
    r = ExchangeSimulator(strategy=strat, config=cfg).run(Path(fold['path']))
    a = np.array(rec, dtype=np.int64).reshape(-1, 3)
    return {'fold': fold['name'], 'pnl': r.pnl, 'fills': r.fills, 'q50': r.abs_inventory_q50,
            'q95': r.abs_inventory_q95, 'breach': r.breach_penalty, 'inventory': r.inventory,
            'cash': r.cash, 'mark_price': r.mark_price,
            't0_us': fold['trade_start_us'], 't1_us': int(round(fold['end_s'] * 1e6)),
            'f_t': a[:, 0], 'f_q': a[:, 1], 'f_px': a[:, 2]}


def decompose(r: dict, smooth_s: float = 60.0) -> dict:
    """Split fold PnL into: spread capture at the fill (vs tape mid), inventory PnL (sum pos * dmid),
    and end liquidation (mark vs mid). Inventory PnL is further split into a slow 'directional' part
    (pos averaged over smooth_s blocks times the block's mid change) and the fast remainder
    (adverse selection / short-term mean reversion)."""
    t, q, px = r['f_t'], r['f_q'], r['f_px']
    t0, t1 = r['t0_us'], r['t1_us']
    m_fill = mid_at(t)
    spread = float(np.sum(q * (m_fill - px)))
    m_end = float(mid_at(np.array([t1]))[0])
    inv = int(q.sum())
    liq = r['pnl'] - (-(q * px).sum() + inv * m_end) if len(q) else 0.0  # mark value - mid value
    # inventory pnl on an event grid: pos is piecewise-constant between fills
    pos_after = np.cumsum(q)
    # sum over fills of pos_before*(mid at fill - mid at previous fill) + final pos*(m_end - m_last)
    mids = np.concatenate([[mid_at(np.array([t0]))[0]], m_fill, [m_end]])
    pos_seg = np.concatenate([[0], pos_after])
    inv_pnl = float(np.sum(pos_seg * np.diff(mids)))
    # time-weighted position, 1 s grid
    grid = np.arange(t0, t1, 1_000_000)
    pos_g = np.concatenate([[0], pos_after])[np.searchsorted(t, grid, side='right')]
    mid_g = mid_at(grid)
    dm = np.diff(np.concatenate([mid_g, [m_end]]))
    k = max(1, int(smooth_s))
    # block-level: mean pos in a smooth_s block * the block's mid change
    edges = np.append(mid_g[::k], m_end)
    blocks = [pos_g[i:i + k].mean() for i in range(0, len(grid), k)]
    slow = float(np.sum(np.array(blocks) * np.diff(edges)))
    whole = float(pos_g.mean() * (m_end - mid_g[0]))
    a = np.abs(pos_g)
    return {
        'pnl': r['pnl'], 'spread': spread, 'inv_pnl': inv_pnl, 'liq': liq,
        'resid': r['pnl'] - spread - inv_pnl - liq,
        'slow': slow, 'fast': inv_pnl - slow, 'whole_drift': whole,
        'mean_pos': float(pos_g.mean()), 'dmid': m_end - mid_g[0],
        'pin100': float((a >= 100).mean()), 'pin95': float((a >= 95).mean()),
        'long95': float((pos_g >= 95).mean()), 'short95': float((pos_g <= -95).mean()),
        'abs_mean': float(a.mean()), 'vol': int(np.abs(q).sum()), 'end_inv': inv,
    }
