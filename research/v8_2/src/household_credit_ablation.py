"""Credit policy counterfactual: expands households' bank loans WITHOUT manually setting M2."""
from pathlib import Path
import sys,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_v82 import runone,ROOT
rows=[]
for seed in range(10100,10104):
 for stress in (False,True):
  for mode in ['capitalism','russia_mixed','soviet_planning','synergy_cyber']:
   for policy in ('baseline','escrow_demand'):
    for cr in (0.,.01):
     d=runone(mode,policy,seed,stress,cr,.016)
     z=d.iloc[119];o=d.iloc[83]
     rows.append(dict(seed=seed,stress=int(stress),mode=mode,policy=policy,hh_credit_rate=cr,
       m2_factor=float(z.m2_proxy/o.m2_proxy),household_loans_end=float(z.household_loans),
       real_sales_oos=float(z.real_final_sales-o.real_final_sales),bank_capital=float(z.bank_min_capratio),
       banks=int(z.bank_resolutions-o.bank_resolutions)))
pd.DataFrame(rows).to_csv(ROOT/'results/household_credit_activation_128_runs.csv',index=False)