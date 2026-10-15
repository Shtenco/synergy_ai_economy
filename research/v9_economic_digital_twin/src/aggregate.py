from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from twin import MODES,POLICIES
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';F=ROOT/'figures'
all_runs=pd.concat([pd.read_csv(R/f'batch_{m}.csv') for m in MODES],ignore_index=True)
all_runs.to_csv(R/'all_runs.csv',index=False)
months=pd.concat([pd.read_csv(R/f'path_{m}.csv') for m in MODES],ignore_index=True)
months.to_csv(R/'representative_paths.csv',index=False)
group=all_runs.groupby(['mode','stress','policy'],as_index=False).agg(
  n=('seed','size'),mean_oos_output=('oos_real_output','mean'),mean_oos_consumption=('oos_consumption','mean'),
  mean_projects=('oos_projects','mean'),mean_abandoned=('oos_abandoned','mean'),
  mean_written_off=('oos_writeoff','mean'),mean_trade=('oos_trade_value','mean'),
  mean_escrow=('oos_escrow','mean'),mean_fiscal=('oos_fiscal','mean'),mean_gini=('household_gini','mean'),
  mean_money=('model_deposits','mean'))
group.to_csv(R/'group_means.csv',index=False)
paired=[]
for mode in MODES:
 for stress in (0,1):
  subset=all_runs[(all_runs['mode']==mode)&(all_runs['stress']==stress)]
  base=subset[subset['policy']=='baseline'].set_index('seed')
  for pol in POLICIES:
   obs=subset[subset['policy']==pol].set_index('seed')
   dy=(obs['oos_real_output']-base['oos_real_output']).to_numpy()
   dc=(obs['oos_consumption']-base['oos_consumption']).to_numpy()
   n=len(dy);t=3.182 if n==4 else 1.96
   em=t*dy.std(ddof=1)/np.sqrt(n);ec=t*dc.std(ddof=1)/np.sqrt(n)
   paired.append(dict(mode=mode,stress=stress,policy=pol,n=n,delta_output=dy.mean(),lo95=dy.mean()-em,hi95=dy.mean()+em,
     delta_cons=dc.mean(),cons_lo95=dc.mean()-ec,cons_hi95=dc.mean()+ec))
pd.DataFrame(paired).to_csv(R/'paired_95ci.csv',index=False)
fig,ax=plt.subplots(figsize=(10,5));subset=group[group.stress==1];labels=['Capitalism','Soviet-like','Mixed Russia','SYNERGY'];x=np.arange(4)
for j,(pol,label) in enumerate([('baseline','No program'),('escrow_demand','Escrow + demand')]):
 vals=[subset[(subset['mode']==m)&(subset['policy']==pol)]['mean_oos_output'].iloc[0] for m in MODES]
 ax.bar(x+(j-.5)*.35,vals,.35,label=label)
ax.set_xticks(x,labels);ax.set_ylabel('Cumulative physical gross output, synthetic units');ax.set_title('2023–2025 scenario window • common shocks • n=4');ax.legend();fig.tight_layout();fig.savefig(F/'01_physical_output.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5))
for j,(pol,label) in enumerate([('baseline','No program'),('escrow_demand','Escrow + demand')]):
 vals=[subset[(subset['mode']==m)&(subset['policy']==pol)]['mean_oos_consumption'].iloc[0] for m in MODES]
 ax.bar(x+(j-.5)*.35,vals,.35,label=label)
ax.set_xticks(x,labels);ax.set_ylabel('Cumulative household goods consumption');ax.set_title('2023–2025 • same physical technologies');ax.legend();fig.tight_layout();fig.savefig(F/'02_consumption.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5));paired_df=pd.DataFrame(paired);r=paired_df[(paired_df.stress==1)&(paired_df.policy=='escrow_demand')]
val=[r[r['mode']==m]['delta_output'].iloc[0] for m in MODES]
lo=[r[r['mode']==m]['lo95'].iloc[0] for m in MODES];hi=[r[r['mode']==m]['hi95'].iloc[0] for m in MODES]
ax.errorbar(x,val,yerr=[np.array(val)-np.array(lo),np.array(hi)-np.array(val)],fmt='o',capsize=6)
ax.axhline(0,linewidth=1);ax.set_xticks(x,labels);ax.set_ylabel('Physical output difference');ax.set_title('Paired effect of escrow + demand (95% t CI, n=4)');fig.tight_layout();fig.savefig(F/'03_program_uncertainty.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5));path=months[(months['mode']=='synergy')&(months['policy']=='escrow_demand')&(months['stress']==1)]
ax.plot(path['month'],path['physical_output'],label='Cumulative physical gross output')
ax.plot(path['month'],path['consumption'],label='Cumulative household consumption')
ax.set_xlabel('Simulation month');ax.set_ylabel('Synthetic physical units');ax.set_title('SYNERGY • a single trajectory, seed 2000');ax.legend();fig.tight_layout();fig.savefig(F/'04_synergy_real_path.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5));
for m in MODES:
 p=months[(months['mode']==m)&(months['policy']=='escrow_demand')&(months['stress']==1)]
 ax.plot(p['month'],p['commissioned'],label=m)
ax.set_xlabel('Simulation month');ax.set_ylabel('Cumulative commissioned projects');ax.set_title('Real project commissioning with input and skill gates');ax.legend();fig.tight_layout();fig.savefig(F/'05_plant_commissioning.png',dpi=160);plt.close(fig)
checks=dict(software='SINERGY V9 bounded 5-country real-goods ABM',scenario_runs=len(all_runs),train_runs=len(pd.read_csv(R/'train_grid.csv')),
  countries=5,goods=8,household_agents=300,sector_firms=40,
  unique_policy_rules=list(MODES),monetary_policies=list(POLICIES),
  max_goods_identity_error=float(all_runs.goods_audit_error.max()),
  max_bank_identity_error=float(all_runs.bank_audit_error.max()),
  completed_runs_all_120_months=True,strict_historical_macro_calibration=False,
  real_rosstat_io_workbook_ingested=False,fx_set='120 month-end observed CBR USD/RUB, lagged only')
(R/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
print(json.dumps(checks,ensure_ascii=False,indent=2))
print('\nCRISIS GROUP MEANS')
print(subset[subset.policy.isin(['baseline','escrow_demand'])][['mode','policy','mean_oos_output','mean_oos_consumption','mean_projects','mean_escrow','mean_fiscal','mean_gini']].to_string(index=False))