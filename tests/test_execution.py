"""Execution diagnostics must honor FIFO, event ordering, censoring, and cancel latency."""
from pathlib import Path

import numpy as np
import pytest

from analysis import execution as E
from analysis.replay import REPLAY_VERSION, replay_csv
from data_explorer.main import build_app


@pytest.fixture()
def tape(tmp_path: Path):
    path = tmp_path / 'execution.csv'
    path.write_text('Time,Operation,OrderId,Side,Volume,Price,Lifespan\n'
                    '0,Insert,b1,B,10,100,G\n'
                    '0,Insert,a1,A,20,102,G\n'
                    '0.001,Insert,b2,B,5,100,G\n'
                    '0.002,Insert,s1,A,6,100,F\n'
                    '0.003,Cancel,b1,,,,\n'
                    '0.004,Insert,s2,A,2,100,F\n'
                    '0.006,Insert,s3,A,3,100,F\n'
                    '0.007,Insert,b3,B,10,99,G\n'
                    '0.010,Insert,buy,B,20,102,F\n'
                    '0.011,Insert,a2,A,10,104,G\n'
                    '0.100,Insert,b4,B,10,98,G\n')
    return replay_csv(path)


def test_queue_tracks_remaining_volume_and_partial_fills(tape):
    assert REPLAY_VERSION >= 3
    b2 = tape.orders.set_index('order_id').loc['b2']
    assert b2.queue_ahead == 10 and b2.resting_quantity == 5
    short, summary = E.queue_fills(tape, horizon_s=.003, side='B', size='1-5')
    assert summary['orders'] == 1
    assert short.iloc[0]['queue'] == '6-20'
    assert short.iloc[0].any_fill == 1
    assert short.iloc[0].full_fill == 0
    assert short.iloc[0].mean_fraction == pytest.approx(.4)
    assert short.iloc[0].median_first_ms == 3
    full, _ = E.queue_fills(tape, horizon_s=.005, side='B', size='1-5')
    assert full.iloc[0].full_fill == 1


def test_cancelled_and_unobserved_orders_are_not_confused(tape):
    # b1 is canceled after a partial fill; it is not a full fill or a censored order.
    stats, summary = E.queue_fills(tape, horizon_s=.01, side='B', size='6-20', placement='all')
    assert summary['excluded_tail'] == 1  # b4, at the very end
    assert summary['orders'] == 2  # b1 and b3
    first_queue = stats.loc[stats['queue'] == '0'].iloc[0]
    assert first_queue.full_fill == 0
    assert first_queue.any_fill == .5
    assert first_queue.mean_fraction == pytest.approx(.3)
    empty, summary = E.queue_fills(tape, window=(.001, .003), horizon_s=.005, size='1-5')
    assert empty.empty and summary['orders'] == 0 and summary['excluded_tail'] == 1


def test_cancel_race_excludes_fill_exactly_at_activation(tape):
    _, cancel, summary = E.latency_risk(tape, latency_ms=3, cancel_age_ms=1,
                                       side='B', size='1-5', sample_s=.001)
    assert summary['cancel_orders'] == 1
    row = cancel.loc[cancel.delay_ms == 3].iloc[0]
    assert row.any_fill == 1 and row.mean_lost_lots == 2 and row.full_fill == 0
    _, exact, _ = E.latency_risk(tape, latency_ms=2, cancel_age_ms=1,
                                 side='B', size='1-5', sample_s=.001)
    assert exact.loc[exact.delay_ms == 2].iloc[0].any_fill == 0
    assert cancel.loc[cancel.delay_ms == 5].iloc[0].full_fill == 1


def test_maker_sign_volume_weighting_and_tail(tape):
    tables, summary = E.maker_markouts(tape, horizon_s=.001)
    assert summary['fills'] == 4
    assert summary['lots'] == 31
    assert summary['mean'] == pytest.approx(19.5 / 31)
    assert summary['adverse'] == 0
    assert 'No history' in tables['volatility']['group'].tolist()
    _, buy = E.maker_markouts(tape, horizon_s=.001, side='B')
    assert buy['mean'] == pytest.approx(9.5 / 11)
    _, tail = E.maker_markouts(tape, window=(0, .005), horizon_s=.002)
    assert tail['excluded_tail'] == 1  # the fill at .004 cannot use prices beyond .005
    assert tail['fills'] == 1


def test_prefill_state_respects_events_with_same_timestamp(tmp_path):
    path = tmp_path / 'ties.csv'
    path.write_text('Time,Operation,OrderId,Side,Volume,Price,Lifespan\n'
                    '0,Insert,b,B,5,100,G\n0,Insert,a,A,5,102,G\n'
                    '0.001,Insert,a2,A,10,103,G\n'
                    '0.001,Insert,t,B,5,102,F\n'
                    '0.010,Insert,b2,B,5,99,G\n')
    t = replay_csv(path)
    tables, _ = E.maker_markouts(t, horizon_s=.001)
    assert tables['spread']['group'].tolist() == ['2']  # not post-fill spread 3
    assert tables['spread'].iloc[0]['mean'] == pytest.approx(.5)  # sell 102, future mid 101.5


def test_new_tabs_and_empty_windows_render(tape):
    app = build_app({tape.name: tape})
    client = app.server.test_client()
    assert client.get('/_dash-layout').status_code == 200
    for tab in ('queue', 'maker', 'latency'):
        for window in (None, [.09, .1]):
            inputs = [('tab', 'value', tab), ('tape', 'value', tape.name), ('window', 'data', window),
                      ('dt', 'value', .001), ('exec-side', 'value', 'both'), ('exec-size', 'value', 'all'),
                      ('exec-placement', 'value', 'touch'), ('exec-horizon', 'value', .001),
                      ('exec-latency', 'value', 5), ('exec-cancel-age', 'value', 1)]
            response = client.post('/_dash-update-component', json={
                'output': 'content.children', 'outputs': {'id': 'content', 'property': 'children'},
                'inputs': [{'id': i, 'property': p, 'value': v} for i, p, v in inputs],
                'state': [], 'changedPropIds': ['tab.value'],
            })
            assert response.status_code == 200, response.data.decode()


def test_quote_lookup_does_not_extend_past_tape_or_hide_missing_side(tape):
    result = E._quote_values(tape, np.array([6000, tape.end_us + 1]))
    assert np.isnan(result).all()
