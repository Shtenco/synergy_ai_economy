"""Restartable all-mode project-on vs instant-investment ablation; no new training."""
import sys,json,argparse
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_v83 import simulate,deltas,OUT
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,default=8);c=ap.parse_args()
 z=pd.read_csv(OUT/'main_4x4x2.csv');w=json.loads((OUT/'freeze_manifest.json').read_text())['w_shortage']
 rows=[]
 for seed in range(9800,9800+c.seeds):
  for stress in (False,True):
   for mode in ('capitalism','soviet_planning','russia_mixed','synergy_cyber'):
    e,d=simulate(mode,'escrow_demand',seed,stress,w,project=False)
    control=deltas(d)
    case=z[(z.seed==seed)&(z.stress==int(stress))&(z['mode']==mode)&(z.policy=='escrow_demand')].iloc[0]
    rows.append(dict(seed=seed,stress=int(stress),mode=mode,with_factory_sales=case.sales,
      without_factory_sales=control['sales'],delta=case.sales-control['sales'],
      with_factory_cons=case.consumption,without_factory_cons=control['consumption'],
      delta_cons=case.consumption-control['consumption']))
  print('Ablation seed',seed,flush=True)
 r=pd.DataFrame(rows);r.to_csv(OUT/'all_modes_project_factory_ablation.csv',index=False)
 audit={'rows_main':len(z),'ablation_rows':len(r),'historical_IO_loaded':False,
  'financial_balance_max':float(z.max_fin_error.max()),'goods_balance_max':float(z.max_io_error.max()),
  'escrow_mapped_to_cofinance':'inherited from V8.2, zero residual',
  'warning':'Comparison is with a less realistic instant-investment benchmark, not an equally detailed alternative project pipeline'}
 (OUT/'experiment_checks.json').write_text(json.dumps(audit,indent=2))
 print('ABLATION',r.groupby(['stress','mode'])[['delta','delta_cons']].mean().round(2).to_string(),flush=True)