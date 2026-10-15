"""V8.2 calibration, paired four-system OOS and IO uncertainty.
All ranking/policy setting TRAIN 2016-22; OOS 2023-25 only for reporting.
No external capital magically appears in simulation.
"""
from pathlib import Path
import argparse,sys,json
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from economy_v82 import Economy82,Config82,MODES,POLICIES
import economy_v81_legacy as legacy
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
BUDGET=pd.read_csv(ROOT/'data/federal_budget_observed_2019_2025.csv').set_index('year')
M2=pd.read_csv(ROOT/'data/m2_secondary_2016_2025.csv').set_index('year')
TARGET=float(M2.loc[2022,'m2_trillion_rub']/M2.loc[2019,'m2_trillion_rub'])
TARGET_OOS=float(M2.loc[2025,'m2_trillion_rub']/M2.loc[2022,'m2_trillion_rub'])
FISCAL_TRAIN=float((BUDGET.loc[2019:2022].expenditure_billion_rub/BUDGET.loc[2019:2022].revenue_billion_rub).mean())

def runone(mode,policy,seed,stress,cr,amort,scale=1.0):
 if scale!=1:
  legacy.A=BASE_A*scale
 try:
  cfg=Config82(mode=mode,policy=policy,seed=seed,stress=stress,response=.4,
   fiscal_scale=7.5,credit_scale=2.,household_credit_rate=cr,household_amortization=amort)
  return Economy82(cfg).run(120)
 finally:
  legacy.A=BASE_A.copy()
BASE_A=legacy.A.copy()

def calibrate(seeds):
 rows=[]
 # Do not use 2023-25 observations or simulator outputs during parameter choice.
 for cr in [0.,.003,.01,.025,.05]:
  for amort in [0.,.008,.016]:
   for seed in range(7000,7000+seeds):
    d=Economy82(Config82(mode='russia_mixed',policy='baseline',seed=seed,
      fiscal_scale=7.5,credit_scale=2.,household_credit_rate=cr,household_amortization=amort)).run(84)
    old=d.iloc[47];end=d.iloc[83]
    factor=float(end.m2_proxy/old.m2_proxy)
    row=dict(credit_rate=cr,amortization=amort,seed=seed,obs_M2_factor=TARGET,model_M2_proxy_factor=factor,
       abs_log_err=abs(np.log(factor/TARGET)),cash_share=float(end.m0_currency/end.m2_proxy),
       bank_min_capratio=float(end.bank_min_capratio),bank_resolutions=int(end.bank_resolutions))
    rows.append(row)
 grid=pd.DataFrame(rows);grid.to_csv(OUT/'m2_train_2019_2022_grid.csv',index=False)
 mean=grid.groupby(['credit_rate','amortization'],as_index=False).mean(numeric_only=True).sort_values('abs_log_err')
 mean.to_csv(OUT/'m2_train_objective.csv',index=False)
 chosen=mean.iloc[0];frozen={
   'train_only':True,'trained_months':'2016-01/2022-12','m2_eval':'2019 end to 2022 end','oos_months':'2023-01/2025-12',
   'credit_rate':float(chosen.credit_rate),'amortization':float(chosen.amortization),
   'fiscal_scale':7.5,'credit_scale':2.0,
   'secondary_observed_m2_factor':TARGET,'model_m2_proxy_factor':float(chosen.model_M2_proxy_factor),
   'm2_miscalibration_absolute_log_error':float(chosen.abs_log_err),
   'pass_m2_growth_10pct_rel_error':bool(abs(chosen.model_M2_proxy_factor/TARGET-1)<.10),
   'sources_note':'M2 historical annual series is secondary reproduction. M2 proxy includes deposits and nonbank cash; sectoral coverage remains incomplete.'}
 (OUT/'train_frozen_v82.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf8')
 return frozen

def eval_main(nseeds,params):
 rows=[];paths=[];b=range(9800,9800+nseeds)
 for ix,seed in enumerate(b):
  for stress in [False,True]:
   for mode in MODES:
    for policy in POLICIES:
     d=runone(mode,policy,seed,stress,params['credit_rate'],params['amortization'])
     old=d.iloc[83];end=d.iloc[119]
     row=dict(seed=seed,mode=mode,policy=policy,stress=int(stress),
      oos_physical_sales=float(end.real_final_sales-old.real_final_sales),
      oos_consumption=float(end.real_consumption-old.real_consumption),
      oos_gross_output=float(end.real_output_gross-old.real_output_gross),
      oos_household_new_loans=float(end.household_credit_issued-old.household_credit_issued),
      bank_resolutions=int(end.bank_resolutions-old.bank_resolutions),
      bank_min_capratio=float(end.bank_min_capratio), hh_gini=float(end.hh_liquid_gini),
      cbr_cash=float(end.m0_currency), m2_proxy=float(end.m2_proxy),
      m2_oos_factor=float(end.m2_proxy/old.m2_proxy),
      m2_oos_obs_factor=TARGET_OOS,
      matched_escrow=float(end.bank_matched-old.bank_matched),
      unlocked_escrow=float(end.escrow_unlocked-old.escrow_unlocked),
      demand_vouchers=float(end.demand_vouchers-old.demand_vouchers),
      govt_debt_growth=float(end.gov_debt-old.gov_debt),
      max_double_entry_error=float(end.financial_residual),
      max_io_physical_error=float(end.input_output_residual),
      end_inflation=float(end.inflation))
     rows.append(row)
     if seed==9800 and stress and policy in ('baseline','escrow_demand'):
      paths.append(d.assign(scenario_mode=mode,scenario_policy=policy,scenario_stress=int(stress)))
  print('V8.2 MAIN',ix+1,'/',len(b),'seed',seed,flush=True)
 r=pd.DataFrame(rows);r.to_csv(OUT/'four_economies_main_runs.csv',index=False)
 g=r.groupby(['stress','mode','policy'],as_index=False).agg(n=('seed','size'),sales=('oos_physical_sales','mean'),consumption=('oos_consumption','mean'),bank_events=('bank_resolutions','mean'),gini=('hh_gini','mean'),bank_capital=('bank_min_capratio','mean'),m2_annual_3yr_factor=('m2_oos_factor','mean'),escrow=('unlocked_escrow','mean'),demand=('demand_vouchers','mean'))
 g.to_csv(OUT/'group_means.csv',index=False)
 if paths:pd.concat(paths,ignore_index=True).to_csv(OUT/'representative_monthly_paths.csv',index=False)
 # Strictly matched paired differences across shared seeds & stress.
 p=r.pivot(index=['seed','stress','mode'],columns='policy',values=['oos_physical_sales','oos_consumption','bank_resolutions'])
 diff=[]
 for (seed,stress,mode),row in p.iterrows():
  diff.append(dict(seed=seed,stress=stress,mode=mode,net_sales=row[('oos_physical_sales','escrow_demand')]-row[('oos_physical_sales','baseline')],
    net_consumption=row[('oos_consumption','escrow_demand')]-row[('oos_consumption','baseline')],
    net_bank_resolutions=row[('bank_resolutions','escrow_demand')]-row[('bank_resolutions','baseline')]))
 pd.DataFrame(diff).to_csv(OUT/'paired_policy_differences.csv',index=False)
 checks=dict(runs=len(r),expected=32*nseeds,finance_max=float(r.max_double_entry_error.max()),io_max=float(r.max_io_physical_error.max()),
   escrow_match_max=float((r.matched_escrow-r.unlocked_escrow).abs().max()),
   all_modes_same_policies=bool((r.groupby(['stress','mode']).policy.nunique()==4).all()),
   historical_rosstat_matrix_loaded=False,m2_train_pass=bool(params['pass_m2_growth_10pct_rel_error']))
 assert checks['finance_max']<1e-6 and checks['io_max']<1e-6 and checks['escrow_match_max']<1e-8 and checks['all_modes_same_policies']
 (OUT/'audit_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
 return r

def io_sensitivity(params,nseeds=3):
 rows=[]
 for sc in [.8,1.,1.2]:
  for seed in range(9900,9900+nseeds):
   for stress in [False,True]:
    for mode in MODES:
     for policy in ['baseline','escrow_demand']:
      d=runone(mode,policy,seed,stress,params['credit_rate'],params['amortization'],sc)
      rows.append(dict(io_scale=sc,seed=seed,stress=int(stress),mode=mode,policy=policy,
       oos_sales=float(d.iloc[-1].real_final_sales-d.iloc[83].real_final_sales),
       oos_consumption=float(d.iloc[-1].real_consumption-d.iloc[83].real_consumption),
       io_error=float(d.iloc[-1].input_output_residual)))
 out=pd.DataFrame(rows);out.to_csv(OUT/'io_illustrative_sensitivity.csv',index=False)
 print('V8.2 IO SENSITIVITY',len(out),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,default=10);ap.add_argument('--calibration-seeds',type=int,default=4);ap.add_argument('--io-seeds',type=int,default=3);v=ap.parse_args()
 p=calibrate(v.calibration_seeds);print('TRAIN FROZEN',p,flush=True)
 r=eval_main(v.seeds,p)
 io_sensitivity(p,v.io_seeds)
 print(r.groupby(['stress','mode','policy'])[['oos_physical_sales','oos_consumption','bank_resolutions','m2_oos_factor']].mean().round(2).to_string(),flush=True)