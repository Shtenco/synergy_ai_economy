import csv,json
from pathlib import Path
import numpy as np
from twin import Twin,Config,MODES
rdir=Path(__file__).resolve().parents[1]/'results'
wgrid=[0.,.25,.5,.75,1.]
rows=[];best={}
for mode in MODES:
 for w in wgrid:
  for j in range(3):
   e=Twin(Config(mode=mode,policy='baseline',seed=1000+j,nmonths=84,planning_weight=w,stress=bool(j%2)))
   e.run();h=e.history[-1]
   obj=.45*h['physical_output']+.65*h['consumption']-.20*h['capex_written_off']-.02*h['gov_debt']
   rows.append(dict(mode=mode,weight=w,train_seed=1000+j,objective=obj,real_output=h['physical_output'],consumption=h['consumption']))
 for k in [mode]:
  avg={w:np.mean([x['objective'] for x in rows if x['mode']==k and x['weight']==w]) for w in wgrid}
  best[k]=min(wgrid,key=lambda w:(-avg[w],w))
with (rdir/'train_grid.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
(rdir/'frozen_policies.json').write_text(json.dumps({'train_seeds':[1000,1001,1002], 'training_months':84,'oos_months':36,'best_policy_weight':best,
 'all_modes_same_candidate_grid':wgrid,'all_modes_same_objective':'0.45*output + 0.65*consumption - 0.20*writeoff - 0.02*debt',
 'real_macro_calibration':False,'only_historical_observed_input':'monthly_USDRUB_2016_2025'},indent=2),encoding='utf-8')
print('TRAIN',len(rows),'FREEZE',best)