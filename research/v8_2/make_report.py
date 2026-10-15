from pathlib import Path
import numpy as np,pandas as pd,json,hashlib,zipfile
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;R=ROOT/'results';G=ROOT/'figures';G.mkdir(exist_ok=True)
d=pd.read_csv(R/'four_economies_main_runs.csv');ps=pd.read_csv(R/'paired_policy_differences.csv');p=json.loads((R/'train_frozen_v82.json').read_text());checks=json.loads((R/'audit_checks.json').read_text());
ci=[]
for (st,mode),x in ps.groupby(['stress','mode']):
 row=dict(stress=st,mode=mode,n=len(x))
 for a in ['net_sales','net_consumption','net_bank_resolutions']:
  z=x[a].values;mean=z.mean();hw=t.ppf(.975,len(z)-1)*z.std(ddof=1)/np.sqrt(len(z)) if len(z)>1 else float('nan');row[a+'_mean']=float(mean);row[a+'_CI_lo']=float(mean-hw);row[a+'_CI_hi']=float(mean+hw);row[a+'_positive_seeds']=int(sum(z>0))
 ci.append(row)
pd.DataFrame(ci).to_csv(R/'paired_95CI.csv',index=False)
scenarios=d[(d.stress==1)&d.policy.isin(['baseline','escrow_demand'])]
pivot=scenarios.pivot_table(index='mode',columns='policy',values='oos_physical_sales',aggfunc='mean')
fig,ax=plt.subplots(figsize=(11,5));idx=np.arange(len(pivot));ax.bar(idx-.18,pivot['baseline'],width=.36,label='Baseline');ax.bar(idx+.18,pivot['escrow_demand'],width=.36,label='Escrow + demand');ax.set_xticks(idx,[x.replace('_',' ') for x in pivot.index]);ax.set_ylabel('Synthetic final physical output, OOS 2023-25');ax.set_title('V8.2: Four systems under shared shocks (12 seeds)');ax.legend();fig.tight_layout();fig.savefig(G/'crisis_four_economies.png',dpi=175);plt.close(fig)
a=pd.read_csv(R/'m2_train_objective.csv')
fig,ax=plt.subplots(figsize=(9,4));ax.bar(['Observed proxy growth 2019-22','Best modeled M2-proxy'],[p['secondary_observed_m2_factor'],p['model_m2_proxy_factor']]);ax.set_ylabel('End-of-2022 / End-of-2019');ax.set_title('Failed monetary calibration: even with modeled currency cash');fig.tight_layout();fig.savefig(G/'m2_calibration_failure.png',dpi=175);plt.close(fig)
# Physical IO uncertainty, scale 0.8, 1.0, 1.2
io=pd.read_csv(R/'io_illustrative_sensitivity.csv');x=io[(io.stress==1)&(io.policy=='baseline')].groupby(['mode','io_scale']).oos_sales.mean().unstack();ax=x.T.plot(marker='o',figsize=(9,4));ax.set_xlabel('Illustrative intermediate input coefficient multiplier');ax.set_ylabel('Synthetic physical output');ax.set_title('V8.2 uncertainty: not Rosstat-calibrated!');plt.tight_layout();plt.savefig(G/'io_uncertainty.png',dpi=175);plt.close()
manifest=[]
for f in sorted(ROOT.rglob('*')):
 if f.is_file() and f.name not in {'MANIFEST_SHA256.csv'}:
  manifest.append(dict(file=f.relative_to(ROOT).as_posix(),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
pd.DataFrame(manifest).to_csv(ROOT/'MANIFEST_SHA256.csv',index=False)
print('CI STRESS')
print(pd.DataFrame(ci).query('stress==1')[['mode','net_sales_mean','net_consumption_mean','net_sales_CI_lo','net_sales_CI_hi','net_sales_positive_seeds']].round(3).to_string(index=False))