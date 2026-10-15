"""Uncertainty experiment: IO coefficients are NOT from Rosstat. Uniform +/-20% exposure."""
import sys, json
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
import economy_v81 as em
ROOT=Path(__file__).resolve().parents[1]
p=json.loads((ROOT/'results/train_only_frozen_calibration.json').read_text())
reference=em.A.copy(); rows=[]
for intensity in [.8,1,1.2]:
    em.A=reference*intensity
    for seed in [9800,9801,9802]:
        for mode in em.MODES:
            for policy in ['baseline','escrow_demand']:
                z=em.Economy(em.Config(mode,policy,seed,True,.4,p['fiscal_scale'],p['credit_scale'])).run(120)
                a=z.iloc[83];b=z.iloc[-1]
                rows.append(dict(io_scale=intensity,seed=seed,mode=mode,policy=policy,oos_sales=b.real_final_sales-a.real_final_sales,oos_consumption=b.real_consumption-a.real_consumption,banks=b.bank_resolutions,io_error=b.input_output_residual,financial_error=b.financial_residual))
    print('IO SCALE',intensity,flush=True)
em.A=reference
r=pd.DataFrame(rows);r.to_csv(ROOT/'results/io_uncertainty_72_runs.csv',index=False)
g=r.groupby(['io_scale','mode','policy'])[['oos_sales','oos_consumption','banks']].mean().reset_index()
g.to_csv(ROOT/'results/io_uncertainty_group_means.csv',index=False)
for intensity in [.8,1,1.2]:
    w=g[(g.io_scale==intensity)&(g.policy=='baseline')].sort_values('oos_sales',ascending=False)
    print('IO',intensity,'RANKING',w[['mode','oos_sales']].to_dict('records'),flush=True)
assert r.financial_error.max()<1e-5 and r.io_error.max()<1e-6