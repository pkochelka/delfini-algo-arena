import json
B=None; rows=[]
for l in open('marek/research/round2/hyp/results.jsonl'):
    try: d=json.loads(l)
    except: continue
    if d.get('tag')=='overfit_base': B=d
    elif str(d.get('tag','')).startswith('overfit_'): rows.append(d)
def f(d):
    return d['pnl']['r2'], d['r2_ex5398'], d['r2_sh5398'], d['synth_mean']
b=f(B)
print('| setting | r2 | d r2 | d ex5398 | d 5398 win | d synth mean | synth tapes up/dn | d(ex5398+7*synth_mean) | win_t |')
print('|---|---|---|---|---|---|---|---|---|')
print(f'| final3 base | {b[0]:,.0f} | 0 | 0 (={b[1]:,}) | 0 (={b[2]:,}) | 0 (={b[3]:,.0f}) | | 0 | |')
out=[]
for d in rows:
    x=f(d); dd=[x[i]-b[i] for i in range(4)]
    up=sum(1 for t in d['pnl'] if t!='r2' and d['pnl'][t]>B['pnl'][t]); dn=sum(1 for t in d['pnl'] if t!='r2' and d['pnl'][t]<B['pnl'][t])
    obj=dd[1]+7*dd[3]
    out.append((obj,f"| {' '.join(f'{k}={v}' for k,v in d['set'].items())} | {x[0]:,.0f} | {dd[0]:+,.0f} | {dd[1]:+,.0f} | {dd[2]:+,.0f} | {dd[3]:+,.0f} | {up}/{dn} | {obj:+,.0f} | {d.get('vs_base',{}).get('win_t')} |", dd[0]))
print('\nsorted by full r2 delta:'); [print(o[1]) for o in sorted(out,key=lambda o:-o[2])]
print('\nsorted by 5398-free objective (d ex5398 + 7*d synth_mean = d total pnl on r2-ex + s1..s7):'); [print(o[1]) for o in sorted(out,key=lambda o:-o[0])]
