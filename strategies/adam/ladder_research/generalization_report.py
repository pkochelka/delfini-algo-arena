"""Summarize the completed audit; no model selection or tuning."""
import csv
import html
import json
import statistics
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT=Path(__file__).resolve().parent/'generalization'
outcome=json.loads((ROOT/'outcome.json').read_text())
records=outcome['records']
regimes=json.loads((ROOT/'market_regimes.json').read_text())
def get(label,window):return next(r for r in records if r['label']==label and r['window']==window)
folds=[f'wf{i:02}' for i in range(1,7)]+['test']
neighbors=list(outcome['protocol']['neighbors'])
stress_names=['ioc_half','ioc_double','ioc_cap500','drop_half_buy_ioc','drop_half_sell_ioc']
synthetic=[f'synthetic_{s}' for s in outcome['protocol']['synthetic_seeds']]

fields=['label','window','pnl','pnl_per_hour','traded_volume','pnl_per_unit','abs_inventory_q95','max_abs_inventory','max_drawdown','breach_penalty']
with (ROOT/'comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields)
    w.writeheader()
    for r in records:
        w.writerow({k:(r['pnl']/r['traded_volume'] if k=='pnl_per_unit' else r[k]) for k in fields})

native_rates=[get('native',w)['pnl_per_hour'] for w in folds]
ratios=[]
for label in neighbors:
    ratio=sum(get(label,w)['pnl'] for w in ('wf05','wf06'))/sum(get('native',w)['pnl'] for w in ('wf05','wf06'))
    ratios.append((label,ratio))

lines=['# Does Peter\'s ladder generalize?','',
    '## Assessment','',
    'The evidence supports a profitable rule across the observed round1 regime, rather than a profit confined to one time window. It does not establish generalization to unseen organizer data or different market regimes. The older sample is a separate available stress tape, not a genuinely blind validation sample. Fresh synthetic seeds check dependence on one random realization only under an approximate fitted generator.','',
    '## Frozen implementation and protocol','',
    'The current source was copied to ladder_frozen.py before tests. Current defaults use size_top=400, cap=100, levels=1, topup=10, imb_pull=0; the new imbalance filter is disabled. The previous defaults used size_top=200. With cap 100, each order is clipped to at most 200, but the larger target can trigger small topups differently. Current native earns 107,559 vs the previous 107,532 (+27, or 0.025%); a size_top=200 control exactly reproduces 107,532. The previously selected tuned settings exactly reproduce 22,254.','',
    'The native configuration and the already-selected tuned configuration were held fixed. No parameter was chosen or changed based on this audit. All 51 runs use the official simulator matching engine at the competition\'s 5 ms latency, penalty settings and inventory limits; all had zero penalties and stayed within their strategy cap. A streaming audit subclass records inventory and order statistics without modifying matching. The source file is unchanged.','',
    'The seven chronological slices use Peter\'s existing seeded splits, 60 seconds without trading and fresh zero cash/inventory per slice. First four slices were used for earlier fitting, next two for earlier validation, and the final slice for earlier testing. They are non-overlapping trading periods of one session, not seven independent trading days. Results across them are diagnostic because the public tape has been repeatedly inspected.','',
    '## Temporal stability','',
    '| Window | Native PnL | Native PnL/hour | Tuned PnL | Tuned PnL/hour |',
    '|---|---:|---:|---:|---:|']
for w in folds:
    a,b=get('native',w),get('tuned',w)
    lines.append(f"| {w} | {a['pnl']:,.0f} | {a['pnl_per_hour']:,.0f} | {b['pnl']:,.0f} | {b['pnl_per_hour']:,.0f} |")
lines+=['',f"Native is profitable in {sum(r>0 for r in native_rates)}/7 periods, with PnL/hour ranging {min(native_rates):,.0f}–{max(native_rates):,.0f}.",
    '', '## Nearby native parameters','',
    'One-at-a-time changes, evaluated on wf05 and wf06. Only the named parameter changes; no winning value is selected.','',
    '| Change | Combined PnL / default PnL | Positive windows |',
    '|---|---:|---:|']
for label,ratio in ratios:
    lines.append(f"| {label} | {ratio:.1%} | {sum(get(label,w)['pnl']>0 for w in ('wf05','wf06'))}/2 |")
lines+=['',
    'The active changes (smaller targets and disabling topups) remain profitable, which argues against a narrow parameter peak. Positive topup thresholds 5–40 produce identical results because size_top 400 is well above the per-side room allowed by cap 100, so those thresholds do not bind; that flatness is mechanical, not four independent robustness successes. Size 200 and 400 have the same inventory ceiling but differ slightly in topup behavior. These checks cannot remove bias from the original choice of strategy family.','',
    '## Other tape and fresh synthetic realizations','',
    '| Tape | Native PnL/hour | Tuned PnL/hour |',
    '|---|---:|---:|']
for w in ['legacy_sample']+synthetic:
    lines.append(f"| {w} | {get('native',w)['pnl_per_hour']:,.0f} | {get('tuned',w)['pnl_per_hour']:,.0f} |")
lines+=['',
    'Synthetic tapes use independent, pre-recorded seeds 7101–7105, with 1260 seconds generated and the first 60 seconds excluded from trading. Fresh exponential event clocks, generated lognormal IOC sizes and resampled limit-order marks produce new paths. They do not replay round1 event times or prices.','',
    'The generator was fitted to round1, and earlier diagnostics found about 45% too little one-second volatility and 81% too-favorable maker markouts. Positive synthetic PnL therefore supports survival of fresh draws from this approximate model, not survival of the true unknown distribution. Do not average synthetic and real PnLs into one generalization estimate.','',
    '### Observed real-market differences','',
    '| Tape | IOC/s | Mean IOC size | Mean spread | Mean touch depth | 1s mid-change SD |',
    '|---|---:|---:|---:|---:|---:|']
for r in regimes:
    lines.append(f"| {r['name']} | {r['ioc_per_second']:.2f} | {r['mean_ioc_size']:.1f} | {r['mean_spread']:.2f} | {r['mean_touch_quantity']:.1f} | {r['sigma_1s_ticks']:.3f} |")
lines+=['',
    'Quote statistics use a 100 ms grid offset by 50 ms, after the same 60-second warmup. The 1s statistic is the SD of differences of one-second-spaced mid samples (not annualized). This avoids sampling the legacy tape exactly on its scheduled cancellation timestamps.','',
    '## Structural stress tests','',
    'Same 49-minute final interval; change only IOC rows, preserve all original limit-order messages, cancellation times and prices. These are counterfactual stress replays, not independently regenerated markets: changing executions without changing later order decisions is only an approximation. Directional thinning drops 50% of one IOC side using a fixed seed, rather than changing its price to manufacture crossings.','',
    '| Stress | Native PnL | Native / original | Tuned PnL |',
    '|---|---:|---:|---:|']
for w in stress_names:
    a,b=get('native',w),get('tuned',w)
    lines.append(f"| {w} | {a['pnl']:,.0f} | {a['pnl']/get('native','test')['pnl']:.1%} | {b['pnl']:,.0f} |")
lines+=['',
    '## Conditional uncertainty and limits','',
    'For each run, record three thousand circular moving-block bootstrap replicates of one-minute PnL increments, using five-minute blocks. This preserves some short-run dependence. Reported intervals concern the mean PnL rate conditional on that tape/regime, omit incomplete minutes and do not include model-selection, source-version, generator or regime uncertainty. Seven slices and five synthetic seeds are too few to estimate rare blow-ups reliably.','',
    '| Configuration | Final real-test PnL/hour | Conditional block-bootstrap 95% interval |',
    '|---|---:|---|']
for label in ('native','tuned'):
    r=get(label,'test')
    lo,hi=r['block_bootstrap_95_pnl_per_hour']
    lines.append(f"| {label} | {r['pnl_per_hour']:,.0f} | {lo:,.0f} to {hi:,.0f} |")
lines+=['',
    'The mechanism is plausible for a queue-priority maker: large uninformed aggressive orders reach its resting quotes, spread capture pays, and retaining live orders avoids repeatedly losing queue priority. This depends on the joint behavior of fills, future prices and cancellations, not merely matching the marginal order-size distribution. High inventory-limit occupancy means changes in adverse selection or persistent directional order flow matter. The strategy has no hardcoded future time/price path and reads only the current/public book and its private fills. That rules out an obvious lookahead mechanism, not broader research overfitting.','',
    'A stronger generalization test needs additional organizer tapes or genuinely independent days generated by the same unknown process. Freeze the code and parameters before seeing them; evaluate without retuning; check maker markouts, drawdowns, profitability and time at the inventory limit, including stressed/volatile periods. Any subsequent retuning needs another untouched evaluation set.','',
    '## Reproduce','',
    'From the repository:', '```powershell',
    '.\\.venv\\Scripts\\python.exe -X utf8 -B strategies\\adam\\ladder_research\\generalization.py',
    '.\\.venv\\Scripts\\python.exe -X utf8 -B strategies\\adam\\ladder_research\\generalization_context.py',
    '.\\.venv\\Scripts\\python.exe -X utf8 -B strategies\\adam\\ladder_research\\generalization_report.py',
    '```','',
    'protocol.json records the frozen source hash and pre-specified test grid; outcome.json and individual JSONs include all audited results and conditional intervals; comparison.csv is a compact numeric export. No new candidate or final submission is selected by this work.','']
(ROOT/'report.md').write_text('\n'.join(lines),encoding='utf-8')

fig=make_subplots(rows=2,cols=2,subplot_titles=['Chronological PnL/hour','Nearby native parameters: relative PnL',
    'Fresh synthetic seeds: PnL/hour','Order-flow stress: PnL relative to unchanged tape'])
colors={'native':'#2d75cc','tuned':'#e48a33'}
for label in ('native','tuned'):
    fig.add_bar(x=folds,y=[get(label,w)['pnl_per_hour'] for w in folds],name=label,legendgroup=label,marker_color=colors[label],row=1,col=1)
    fig.add_scatter(x=synthetic,y=[get(label,w)['pnl_per_hour'] for w in synthetic],mode='lines+markers',name=label,legendgroup=label,showlegend=False,line_color=colors[label],row=2,col=1)
    fig.add_bar(x=stress_names,y=[get(label,w)['pnl']/get(label,'test')['pnl'] for w in stress_names],name=label,legendgroup=label,showlegend=False,marker_color=colors[label],row=2,col=2)
fig.add_bar(x=[x for x,y in ratios],y=[y for x,y in ratios],name='relative native PnL',showlegend=False,marker_color='#2d75cc',row=1,col=2)
fig.add_hline(y=1,line_dash='dot',row=1,col=2)
fig.add_hline(y=1,line_dash='dot',row=2,col=2)
fig.update_layout(height=900,template='plotly_white',barmode='group',title='Ladder robustness audit — frozen settings; no retuning',margin=dict(t=100,b=100))
fig.update_yaxes(tickformat='.0%',row=1,col=2)
fig.update_yaxes(tickformat='.0%',row=2,col=2)
# Include the complete readable report without a Markdown runtime dependency.
body=html.escape('\n'.join(lines))
doc='<html><head><meta charset="utf-8"><title>Ladder robustness</title></head><body style="font:15px system-ui;margin:24px">'+fig.to_html(full_html=False,include_plotlyjs=True)+'<h2>Detailed results and limitations</h2><pre style="white-space:pre-wrap;line-height:1.6">'+body+'</pre></body></html>'
(ROOT/'report.html').write_text(doc,encoding='utf-8')
print('Wrote report.md, report.html and comparison.csv.')
print(json.dumps(dict(native_temporal_rates=native_rates,neighbor_ratios=ratios,
    sample={label:get(label,'legacy_sample')['pnl'] for label in ('native','tuned')},
    synthetic_native_rates=[get('native',w)['pnl_per_hour'] for w in synthetic],
    stresses={w:get('native',w)['pnl'] for w in stress_names}),indent=2))
