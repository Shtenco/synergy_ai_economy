import sys,csv,json,argparse
from pathlib import Path
from twin import Config,Twin,POLICIES,MODES
p=argparse.ArgumentParser();p.add_argument('mode',choices=MODES);a=p.parse_args()
root=Path(__file__).resolve().parents[1];weights=json.loads((root/'results/frozen_policies.json').read_text())['best_policy_weight']
rows=[];paths=[]
for policy in POLICIES:
 for stress in (False,True):
  for seed in (2000,2001,2002,2003):
   e=Twin(Config(mode=a.mode,policy=policy,seed=seed,nmonths=120,stress=stress,planning_weight=weights[a.mode]))
   e.run();st=e.history[83];fn=e.history[-1]
   rows.append(dict(mode=a.mode,policy=policy,stress=int(stress),seed=seed,weight=weights[a.mode],
     oos_real_output=fn['physical_output']-st['physical_output'],
     oos_consumption=fn['consumption']-st['consumption'],
     oos_projects=fn['commissioned']-st['commissioned'],
     oos_abandoned=fn['abandoned']-st['abandoned'],
     oos_writeoff=fn['capex_written_off']-st['capex_written_off'],
     oos_trade_value=fn['trade_value']-st['trade_value'],
     oos_escrow=fn['escrow_released']-st['escrow_released'],
     oos_fiscal=fn['fiscal']-st['fiscal'],
     household_gini=fn['household_wealth_gini'],
     model_deposits=fn['m2_proxy'],
     bank_audit_error=fn['bank_audit_error'],goods_audit_error=fn['physical_audit_error']))
   if seed==2000:paths.extend(e.history)
rd=root/'results'
for name,data in [('batch_'+a.mode+'.csv',rows),('path_'+a.mode+'.csv',paths)]:
 with (rd/name).open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,data[0]);w.writeheader();w.writerows(data)
print(a.mode,len(rows),'output mean',round(sum(x['oos_real_output'] for x in rows)/len(rows),1),'max error',max(x['goods_audit_error'] for x in rows))