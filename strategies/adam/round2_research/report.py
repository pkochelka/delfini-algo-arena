"""Readable evidence, model specification and interactive research figures."""
import ast
import hashlib
import html
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from extract import HERE,OUT,GEN,SOURCE,HORIZONS,dump

def read(name):return json.loads((OUT/name).read_text())
def table(frame,columns=None):
    f=frame[columns] if columns else frame
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:,.4g}'
        return str(v)
    return ['| '+' | '.join(map(str,f.columns))+' |','| '+' | '.join(['---']*len(f.columns))+' |']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in f.itertuples(index=False,name=None)]

def main():
    diag=read('train_diagnostics.json');model=read('model_results.json');risk=read('risk_results.json');quality=read('data_quality_and_aggressors.json')
    baseline=pd.DataFrame(read('baseline_results.json'));schema=read('feature_schema.json');unc=read('uncertainty.json')
    price=pd.read_csv(OUT/'price_models.csv');tail=pd.read_csv(OUT/'return_tails.csv');jump=pd.read_csv(OUT/'jump_models.csv')
    qrisk=pd.read_csv(OUT/'tail_risk_models.csv');clusters=pd.read_csv(OUT/'regime_clusters.csv')
    cohorts=pd.read_csv(OUT/'prior_risk_maker_markouts.csv');arrival=pd.read_csv(OUT/'arrival_intensity_models.csv')
    qfill=pd.read_csv(OUT/'queue_1s.csv');flow=read('flow_structure.json');selected=model['lock']['selections']
    chosen=price[(price.split=='test')&(price.selected==True)]
    marks=pd.DataFrame([dict(horizon_s=h,**diag[f'maker_markout_{h:g}s']) for h in (.005,.05,.1,1.,5.)])
    lines=['# Round 2: market structure, predictive information and trading implications','',
        '## Main conclusions','',
        'The useful complexity here is in conditional execution and risk control. A short-horizon direction model generalizes within this public session, while longer-horizon directional models fail the validation comparison. A model that predicts moves can still lose money crossing the spread. No new profitable trading strategy is established by these prediction results.','',
        '- The unchanged Round1 native ladder loses 136,037 in validation and 62,763 in the final test (19m50s each), with zero penalties. The failure is not an inventory-limit penalty. The smaller tuned ladder also loses in both periods. Adaptive spread loses 6,656 in validation and earns 6,662 in test; this mixed result is not robust proof of profitability.',
        '- Round2 order-size marginals resemble Round1: fitting-period IOC mean 160.42, median 61, q99 about 1,501. But future price risk and fill-conditioned outcomes are much worse. Marginal size matching is insufficient for selecting strategies or validating a data generator.',
        '- Mean historical maker markout declines from +0.471 ticks at 5ms to +0.216 at 100ms and −0.259 at 1s. These are signed future-mid minus actual fill-price, volume weighted. They include initial spread capture. Fast positive markouts do not imply profitable aggressive liquidation: paying the exit half-spread can consume that edge.',
        '- The largest 1% of one-second endpoint moves contribute 77.9% of squared movement in the final period. Mean/variance or Gaussian assumptions alone are fragile; rare jumps matter heavily.',
        '- The locked 100ms HGB model reaches test correlation 0.328, R² 0.107 and 76.3% correct direction CONDITIONAL ON a nonzero move. Only 21.5% of these targets move. This is not 76.3% accuracy over every decision or an achievable trading win rate.',
        '- Five-millisecond entry latency and bid/ask crossing remove the apparent easy trading edge. The fixed crossing-opportunity screens have negative mean test results, including high-score subsets. No screen was optimized on test.',
        '- Historical passive-order fill prediction generalizes: selected HGB AUC 0.751. Conditional first-fill markout is harder: selected ridge R² only 0.011, correlation 0.127. The fill model estimates behavior under historical cancellation policy, not the fill probability of arbitrary new quotes.',
        '- Magnitude is more predictable than longer-horizon direction: the one-second ≥5-tick endpoint-move classifier reaches test AUC 0.943 and lowers Brier loss from 0.0702 to 0.0467. AUC measures ranking; it does not guarantee calibrated probabilities or profitable quote suppression.',
        '- Historical fills preceded by intermediate/high risk states often have much worse 1s markouts. The extreme top bucket is small and reverses sign in test. Do not fit a monotonic cancellation rule or choose its cutoff from these test cohorts.',
        '- Executed-aggressor arrival intensity is also forecastable. A lag-only nonlinear Poisson model reduces test deviance from 1.720 to 1.404 (18.4%) with correlation 0.437. A fixed clock-phase model provides little benefit. Prefer adaptive flow estimates over hardcoded session timing.','',
        '## Data and evaluation design','',
        f"Source: data/round2_public.csv, {quality['events']:,} events, {quality['orders']:,} inserts and {quality['matches']:,} matches over about 6,000 seconds. Data SHA256 is in protocol.json. Timestamps are sorted; duplicate timestamps: {quality['duplicate_timestamps']}; duplicate insert IDs: {quality['duplicate_insert_order_ids']}; nonpositive prices/sizes: {quality['nonpositive_prices']}/{quality['nonpositive_sizes']}.",
        '',
        'Split on time, never shuffled: fit 60–3590s; validation 3600–4790s; final scoring 4800–5990s. Ten-second gaps purge overlapping labels. The first 60s supplies history. Price labels span at most 5.005s; historical-order labels at most 2s. All targets require complete future observations. State/history can carry across fitting boundaries, but targets do not.',
        '',
        'Price decisions use a 100ms grid offset by 50ms. Features are observed at decision t; the price label is mid(t+5ms+h)−mid(t+5ms), with h=100ms, 1s or 5s. Entry/exit crossing diagnostics use actual bid/ask at those times. The 5ms simulator latency is never assumed to be zero.',
        '',
        f"Price row counts: {model['row_counts']}. Historical-order counts: {model['order_row_counts']}. Conditional markout counts: {model['conditional_markout_counts']}.",
        '',
        'The public tape had already been opened descriptively. This is a chronological fitting holdout, not a new blind sample. Supplemental risk, activity and cohort analyses reuse the same final block after the initial model comparison; no new strategy was selected on those test numbers. Independent private data is still required.','',
        '## What a live strategy can actually observe','',
        'The 48 market predictors exclude absolute price, absolute clock phase, order IDs, future cancellation/lifetime information and submitted IOC size. The simulator exposes market orders within 20 ticks, not merely 20 individual orders; depth features use that tick restriction.','',
        '| Feature family | Construction and plausible use |',
        '|---|---|',
        '| Spread, touch size/order imbalance, microprice offset | Fair-value adjustment, touch competition, liquidity fragility |',
        '| Depth imbalance and total depth within 3/5/10/20 ticks | Sweep risk, asymmetric replenishment, size capacity |',
        '| Past returns over 0.1/0.5/1/5/30/60s | Short continuation, reversion and shock persistence |',
        '| Trailing RMS and move frequency over 1/5/30/60s | Conditional spread, exposure limits, jump-risk state |',
        '| Executed volume/sign imbalance over 0.1/0.5/1/5/30s | Aggressive flow direction and intensity |',
        '| Best-quote OFI and update counts over the same windows | Visible replenishment/depletion and cancellation pressure |',
        '| Historical order side, size, price placement and actual queue ahead | Fill-probability and first-fill toxicity models; policy dependence must be respected |','',
        'Depth snapshots are as-of and may be up to 100ms old; tick/event quote and trade features use information through t. Transient one-sided states invalidate current prediction rows. Past-price history carries only already observed prices through those gaps. Raw all-depth insert/cancel information and submitted IOC volumes appear in diagnostics only.','',
        '## Market dynamics, activity and distribution tails','',
        f"Fitting mean spread {diag['summary']['spread_mean']:.3f} ticks; one-tick spread share {diag['summary']['spread_1tick_share']:.1%}; mean touch quantity {diag['summary']['mean_touch_qty']:.1f}; RMS volatility signature at 1s {diag['summary']['vol_1s']:.3f} ticks/sqrt-second. Volatility changes substantially through the session; see five_minute_regimes.csv.",
        '',
        f"Submitted IOC arrivals: {diag['ioc']['rate']:.3f}/s. Limit inserts: {diag['limit']['rate']:.3f}/s. Cancellation-message mean lag {flow['cancel_message_lifetime_s']['mean']:.3f}s, median {flow['cancel_message_lifetime_s']['median']:.3f}s; all fitting cancellations link to an earlier insertion. Message lag is not actual resting lifetime if the order filled earlier.",
        '',
        f"Count Fano factors near one at 10ms grow with aggregation: IOC/limit factors {diag['fano_1s']['ioc']:.2f}/{diag['fano_1s']['limit']:.2f} at 1s and {diag['fano_60s']['ioc']:.2f}/{diag['fano_60s']['limit']:.2f} at 60s, using complete equal-exposure bins only. This is inconsistent with a simple constant-rate Poisson description at every scale. It does not identify whether the cause is changing intensity, self-excitation, common shocks or another dependent mechanism. A Hawkes/Cox explanation needs a separately checked likelihood/residual model.",
        '',
        f"Limit-insert/cancel contemporaneous correlation is {flow['count_correlations'][-1]['limit_cancel']:.4f} at 60s; this does not mean future cancels are perfectly predictable. Cancel lag, current visible order age and quote-state transitions should be modeled explicitly before using such correlations operationally.",
        '',
        'Short-return and trade-sign autocorrelations, variance ratios, volatility signature, intensity spectra and 5-minute summaries are exported. The strongest fitting activity spectral peak is around 256s, but a model using that fixed phase does not improve validation. Do not hardcode a periodic generator from a visual peak.',
        '',
        'Endpoint-return tail summaries (overlapping observations; maxima and kurtosis are descriptive):','']
    lines+=table(tail,['split','horizon_s','median','q95','q99','q999','max','top1pct_squared_move_share'])
    lines+=['',
        'Source-event attribution is in jump_event_sources_train.csv. Large quote changes occur after cancellations, IOC trades and limit insertions. The fitting tape has no crossing G orders, so a jump associated with a G insertion need not be an aggressive trade: restoring a new best quote can move the midpoint. Association with an event is not evidence of the unobserved fundamental cause.',
        '',
        '## Directional models and conditional uncertainty','',
        'Fixed candidates: training mean; unscaled microprice residual; standardized ridge alpha=10; histogram gradient boosting with 15 or 31 leaves, 150 iterations, learning rate .05, minimum 100 samples per leaf and L2=10. No time/price trajectory input. Validation MSE chooses each horizon; fitted model bytes and choices are fingerprinted in locked_models.json before test scoring.','']
    lines+=table(chosen,['horizon_s','model','mse','r2','correlation','moving_direction_accuracy','moving_fraction'])
    lines+=['',
        f"A 3,000-replicate bootstrap using 10-second wall-clock blocks gives a conditional 95% interval of {unc['mse_improvement_95'][0]:.1%}–{unc['mse_improvement_95'][1]:.1%} for the 100ms model's MSE reduction versus the training mean. Moving-case direction accuracy interval: {unc['moving_direction_accuracy_95'][0]:.1%}–{unc['moving_direction_accuracy_95'][1]:.1%}. These intervals do not include model-family/research selection or new-regime uncertainty.",
        '',
        'Joint feature-group permutation on validation identifies useful book, print-flow, quote-flow, volatility and return information for the short-horizon model. Permutation can create unrealistic feature combinations and is descriptive importance, not a causal effect. Correlated ridge coefficients are also not individual economic effects.','',
        'The 1s/5s mean forecast was selected because fitted directional models overfit the training relationships and increased validation MSE. Do not substitute a larger model simply because it has more capacity. Possible future work is shrinkage or mixture-of-experts conditional on verified regimes; it needs a new evaluation design.','',
        '### Crossing-cost diagnostic','',
        'Select sign(prediction), then cross at the actual ask/bid after 5ms and mark an opposite-side exit at horizon. Apply fixed score thresholds above the current half-spread plus {0,.25,.5,1} ticks. Thin decisions by the holding horizon. These remain independent hypothetical opportunities, without inventory, fill interaction or actual IOC replay; they are not dollar PnL.','']
    lines+=table(pd.read_csv(OUT/'crossing_opportunities.csv').query("split == 'test' and horizon_s == .1"))
    lines+=['',
        '## Queue fills, adverse selection and cancellation','',
        'Queue tables follow real non-crossing G orders placed at the touch, with actual competing cancellations and complete follow-up. At 1s, any-fill rate drops from about 61.6% with 1–5 units ahead to 17.3% with more than 100 ahead. These are selected historical order cohorts; quote size, price policy and cancellation policy differ between them.','']
    lines+=table(qfill)
    lines+=['',
        'Historical fill model: logistic vs HGB classifier; validation Brier score selects HGB. Test scores:','']
    lines+=table(pd.DataFrame([dict(model=k,**v) for k,v in model['fill_test'].items()]))
    lines+=['',
        'Conditional first-fill markout model: features known before insertion; first fill must occur within 1s; signed markout is measured 1s after that fill. This uses first fills, unweighted per order, unlike the volume-weighted all-match maker table. Consequently its average is not directly comparable to −0.259 ticks above. Validation MSE selects ridge. Test scores:','']
    lines+=table(pd.DataFrame([dict(model=k,**v) for k,v in model['markout_test'].items()]))
    lines+=['',
        'At a hypothetical cancel request age of 100ms, about 0.24% of eligible resting historical orders receive a fill during the next 5ms. Average extra fill quantity is about .091 lots for buys and .069 for sells. These unconditional averages can hide much worse conditional loss during jumps. A cancel is effective after its delay, not when requested.','',
        'Maker markout by horizon, fitting data, signed ticks per unit including initial spread capture:','']
    lines+=table(marks,['horizon_s','lots','mean','p10','median','p90','adverse'])
    lines+=['',
        'Conditional tables by spread, own-side touch share, volatility and aggressor submitted size are included. In fitting data, own-side touch-share >.8 has about −.555 tick 1s maker markout, versus −.079 below .2. This relationship differs from the simple Round1 imbalance narrative. It is descriptive and confounded by other book states; use the multivariate, out-of-time model before writing a one-variable rule.','',
        '## Risk forecasts and changing regimes','',
        'Predict absolute endpoint moves with a 90% quantile HGB; compare fixed training q90 and rolling Gaussian approximations. Predict endpoint-move events |Δmid|≥3 ticks at 100ms, ≥5 at 1s and ≥10 at 5s with a fixed HGB classifier. These are endpoint events, not the probability that the price touches a barrier anywhere inside the interval. Supplemental model selections are locked using validation before test scoring.','']
    lines+=table(jump.query("split == 'test'"),['horizon_s','threshold','model','brier','auc','base_rate','selected'])
    lines+=['','Tail model test coverage and pinball loss (target coverage 90%):','']
    lines+=table(qrisk.query("split == 'test'"),['horizon_s','model','pinball','coverage','mean_prediction','selected'])
    lines+=['',
        'The 1s/5s conditional q90 models cover only 87.7%/88.5% of test absolute moves; they are not reliable 90% bounds without further calibration. Extreme paths can exceed endpoint-based forecasts by more. Keep a deterministic inventory reserve and hard cap regardless of model confidence.','',
        'A three-component Gaussian mixture is fit on current volatility, spread, depth and print activity from the fitting period only. It is a descriptive clustering, not a proven hidden-state model. In test, cluster 0 occupies about 63.5% of observations and has 0.36% ≥5-tick 1s moves; the wider-spread cluster 1 occupies 16.3% and has 31.2%. A single spread/risk setting is unlikely to be appropriate everywhere.','']
    lines+=table(clusters.query("split == 'test'"))
    lines+=['',
        'Historical maker fills grouped by a risk state available at least 5ms before execution; boundaries are the training distribution\'s 50/80/90/95/99 percentiles. No threshold is chosen on test.','']
    lines+=table(cohorts.query("split == 'test'"))
    lines+=['',
        'Risk is not monotonically related to realized maker reward in every bucket: the highest bucket has only 123 test matches and positive mean markout. These cohorts do not simulate cancellations or displaced queue fills. A strategy-level experiment must establish whether risk suppression improves PnL and whether re-entry loses too much queue priority.','',
        '## Transfer of the existing trading strategies','',
        'The following are actual official-simulator replays, with 60s seeded warmup, fixed Round1 parameters, 5ms latency and liquidation-book marking. They do not use the newly trained models. Each validation/test replay starts with zero cash/inventory and trades for 1190 seconds. All have zero penalties and respect their strategy cap.','']
    lines+=table(baseline,['model','split','pnl','traded_volume','max_abs_inventory','abs_inventory_q95','inventory','breach_penalty'])
    lines+=['',
        '## Model architecture to test next','',
        '1. Use the 100ms predictor as a bounded reservation-price/side-selection signal. Shrink it toward zero and require simulated net utility, rather than using its raw sign to cross.',
        '2. Estimate the candidate quote\'s fill probability from queue, order size, placement, public book and recent flow. Model cancellation as a competing outcome. Validate counterfactual quote policies in the official exchange; historical cohorts alone cannot identify their fill probabilities.',
        '3. Estimate signed future markout conditional on a fill, preferably separately for buys/sells and multiple horizons. The present ridge signal is weak. Use a conservative floor or shrinkage rather than treating its output as certain profit.',
        '4. Use conditional tail forecasts and jump probability to set quote size, width and inventory tolerance. Compare an interpretable regime gate against a full nonlinear model. Both must reserve capacity for pending/canceling orders and respect |inventory|≤100 under one-sided fills.',
        '5. Score quote candidates using expected fill reward minus inventory/hedging cost. Preserve queue position when the incremental expected benefit of moving is small; explicitly price the 5ms cancel/re-entry exposure. Do not assume small unconditional cancel-fill probability means small tail loss.',
        '6. Test the integrated model against no-trade and simple conservative baselines in expanding chronological folds. Freeze every choice before a newly reserved test segment; public test reuse in this notebook is not a substitute for private-data validation.',
        '',
        'This is the architecture to investigate, not an optimized or profitable submission. An online feature engine must match the research definitions, filter own quotes correctly, and handle market impact: training used a market-only replay, while the strategy\'s fills can change future book states. The sandbox requires a single file ≤1MiB, no file/network access, and bounded memory/runtime; fitted pickle files are local research artifacts and cannot simply be loaded in a submission. A compact tree/linear representation must be embedded and parity-tested.','',
        '## Reproducibility and limits','',
        'Run from the repository root:','',
        '```powershell',
        '& strategies\adam\round2_research\run_research.ps1',
        '```','',
        'The runner executes extraction, price/fill/markout models, tail/jump/regime models, fixed strategy baselines, interpretation/audits, arrival-intensity modeling, integrity diagnostics and this report. It stops on an error. The original data, public replay helpers and trading strategies are unchanged. Generated feature arrays, fitted estimators, split CSVs and full simulator logs live under generated/ and are ignored by Git. Compact CSV/JSON evidence and source code are shareable.',
        '',
        'Validation checks independently re-index 100 sampled decisions, 300 delayed targets and the corresponding depth timestamps. Source integrity checks pass. These are targeted causality/index audits, not a complete independent simulator-parity test. Historical probability labels are policy-dependent; overlapping targets reduce effective sample size; single-session uncertainty does not cover future regime changes; an AUC or R² gain is not a dollar-profit claim.',
        '',
        'Files: protocol.json / locked_models.json / locked_risk_models.json / locked_flow_model.json document decisions; feature_schema.json and order_schema.json describe arrays; model_results.json and all CSV tables supply exact values; baseline_results.json supplies actual PnL; uncertainty.json and validation.json document checks. The report uses unrounded source values in these files.','']
    (HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    # Interactive figures; exact numeric evidence remains available in the CSVs.
    d=np.load(GEN/'features.npz');names=schema['names'];xx=d['X'];sec=d['seconds'];stride=np.arange(0,len(sec),10)
    figures=[]
    f=make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=['Midprice','Trailing 30s RMS (past observations)'])
    f.add_scatter(x=sec[stride],y=d['mid'][stride],name='mid',row=1,col=1)
    f.add_scatter(x=sec[stride],y=xx[stride,names.index('rms_30s')],name='RMS30',row=2,col=1)
    for x in (3600,4800):f.add_vline(x=x,line_dash='dash',line_color='#999')
    f.update_layout(height=580,title='Round 2 price path and changing risk');figures.append(f)
    f=go.Figure()
    for h in HORIZONS:
        a=price[(price.horizon_s==h)&(price.split=='test')]
        f.add_bar(x=a.model,y=a.r2,name=f'{h:g}s')
    f.update_layout(title='Test R²: complexity helps at 100ms, fails at longer horizons',barmode='group',yaxis_title='R²');figures.append(f)
    f=go.Figure();de=pd.read_csv(OUT/'price_deciles.csv').query("split=='test' and horizon_s==.1")
    f.add_scatter(x=de.pred,y=de.realized,mode='lines+markers',name='observed',customdata=de.n,hovertemplate='pred %{x:.3f}<br>observed %{y:.3f}<br>n %{customdata}<extra></extra>')
    f.add_scatter(x=[de.pred.min(),de.pred.max()],y=[de.pred.min(),de.pred.max()],mode='lines',name='perfect calibration',line_dash='dash')
    f.update_layout(title='100ms test forecast bins: validation-defined boundaries',xaxis_title='Mean forecast (ticks)',yaxis_title='Mean realized delayed return');figures.append(f)
    f=go.Figure(go.Scatter(x=marks.horizon_s,y=marks['mean'],mode='lines+markers'))
    f.update_layout(title='Historical maker edge decays after the fill (fitting data)',xaxis_type='log',xaxis_title='Seconds after fill',yaxis_title='Volume-weighted signed markout (ticks)');f.add_hline(y=0);figures.append(f)
    f=go.Figure(go.Bar(x=qfill.queue,y=qfill.any_fill,customdata=qfill.orders,hovertemplate='queue %{x}<br>fill %{y:.1%}<br>orders %{customdata}<extra></extra>'))
    f.update_layout(title='Queue position and historical fill probability within 1s',yaxis_tickformat='.0%',xaxis_title='Lots ahead at insertion');figures.append(f)
    f=go.Figure()
    for split in ('validation','test'):
        a=cohorts[cohorts.split==split]
        f.add_bar(x=a.training_risk_bucket,y=a.volume_weighted_markout,name=split,customdata=a.prints,hovertemplate='risk bucket %{x}<br>markout %{y:.3f}<br>matches %{customdata}<extra></extra>')
    f.update_layout(title='Prior risk and maker reward: highest test bucket is small',xaxis_title='Training-defined risk bucket',yaxis_title='1s volume-weighted maker markout',barmode='group');figures.append(f)
    f=go.Figure()
    for split in ('validation','test'):
        a=jump[(jump.split==split)&(jump.model=='hgb')]
        f.add_bar(x=a.horizon_s.astype(str),y=a.auc,name=split)
    f.update_layout(title='Absolute endpoint-move classification: AUC',barmode='group',xaxis_title='Horizon (seconds)',yaxis_range=[.5,1]);figures.append(f)
    f=go.Figure()
    for split in ('validation','test'):
        a=baseline[baseline.split==split];f.add_bar(x=a.model,y=a.pnl,name=split)
    f.update_layout(title='Actual transfer backtests: unchanged Round1 strategies',barmode='group',yaxis_title='Simulator PnL');figures.append(f)
    f=go.Figure()
    for split in ('validation','test'):
        a=arrival[arrival.split==split];f.add_bar(x=a.model,y=a.poisson_deviance,name=split)
    f.update_layout(title='Executed-aggressor arrival prediction (lower deviance is better)',barmode='group');figures.append(f)
    # Small Markdown renderer for the detailed document, with tables and code blocks.
    def inline(s):
        s=html.escape(s);return re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
    rendered=[];i=0
    while i<len(lines):
        s=lines[i]
        if s.startswith('```'):
            code=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
            rendered.append('<pre>'+html.escape('\n'.join(code))+'</pre>')
        elif s.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                rows.append(lines[i]);i+=1
            rendered.append('<div class="scroll"><table>')
            for j,row in enumerate(rows):
                if j==1:continue
                tag='th' if j==0 else 'td';cells=row.strip('|').split('|')
                rendered.append('<tr>'+''.join(f'<{tag}>{inline(c.strip())}</{tag}>' for c in cells)+'</tr>')
            rendered.append('</table></div>');continue
        elif s.startswith('#'):
            level=len(s)-len(s.lstrip('#'));rendered.append(f'<h{level}>{inline(s[level:].strip())}</h{level}>')
        elif s:rendered.append('<p>'+inline(s)+'</p>')
        i+=1
    charts=[]
    for i,f in enumerate(figures):
        f.update_layout(template='plotly_white',font_family='Segoe UI, Arial',margin=dict(l=60,r=30,t=65,b=70))
        charts.append('<section>'+f.to_html(full_html=False,include_plotlyjs=i==0,config={'displaylogo':False,'responsive':True})+'</section>')
    css='body{margin:0;background:#f3f6fa;color:#19283b;font:15px/1.65 Segoe UI,Arial}main{max-width:1350px;margin:auto;padding:24px}section,.text{background:white;padding:20px;margin-bottom:20px;border-radius:12px}h1{font-size:28px}h2{margin-top:42px}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:8px;border-bottom:1px solid #dce3eb;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}pre,code{background:#eef2f7}pre{padding:12px;overflow:auto}a{color:#2866b0}'
    doc='<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Round 2 research</title><style>'+css+'</style></head><body><main><h1>Round 2 research: execution, alpha and risk</h1><p>Chronological models, frozen selections, actual baseline PnL. Prediction quality is not a trading-profit claim.</p><p><a href="#details">Detailed findings and methodology</a> · <a href="REPORT.md">Markdown report</a> · <a href="results/price_models.csv">Exact forecast scores</a></p>'+''.join(charts)+'<div class="text" id="details">'+''.join(rendered)+'</div></main></body></html>'
    (HERE/'report.html').write_text(doc,encoding='utf-8')
    for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==read('protocol.json')['data_sha256']
    dump('completion.json',dict(report='REPORT.md',html='report.html',python_sources_parsed=len(list(HERE.glob('*.py'))),
        source_data_unchanged=True,new_submission_created=False,baseline_runs=len(baseline),feature_count=schema['features']))
    print('REPORT COMPLETE',len(lines),'lines; 9 interactive figures.',flush=True)

if __name__=='__main__':main()
