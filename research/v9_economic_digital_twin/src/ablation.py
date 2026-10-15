"""True within-engine ablation: remove Project Factory inspections; preserve all
other physical technology, initial capital, bank/government policy and shocks.
This tests effect of project CAPEX diversion + later supply expansion.
"""
import csv,json,sys
from pathlib import Path
import numpy as np
from twin import Twin,Config,MODES
root=Path(__file__).resolve().parents[1]
sel=json.loads((root/'results/frozen_policies.json').read_text())['best_policy_weight']
rows=[]
for mode in MODES:
 for stress in (False,True):
  for seed in (2000,2001,2002,2003):
   for enable in (False,True):
    e=Twin(Config(mode=mode,seed=seed,policy='escrow_demand',stress=stress,planning_weight=sel[mode],inspection_budget=8 if enable else 0))
    e.run()
    h=e.history[83];z=e.history[-1]
    rows.append(dict(mode=mode,stress=int(stress),seed=seed,project_factory=int(enable),
      oos_output=z['physical_output']-h['physical_output'],
      oos_consumption=z['consumption']-h['consumption'],
      oos_projects=z['commissioned']-h['commissioned'],
      oos_money_creation=z['escrow_released']-h['escrow_released'],
      oos_fiscal=z['fiscal']-h['fiscal']))
with (root/'results/ablation_factory.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
for mode in MODES:
 ss=[]
 for z in (0,1):
  r0={(x['seed']):x for x in rows if x['mode']==mode and x['stress']==z and not x['project_factory']}
  diff=[x['oos_output']-r0[x['seed']]['oos_output'] for x in rows if x['mode']==mode and x['stress']==z and x['project_factory']]
  ss.append((z,round(float(np.mean(diff)),2)))
 print(mode,ss)
print('ablation simulations',len(rows))