"""Train-only calibration and fair four-system replay. IO matrix remains illustrative.
Explicit reject criterion: if deposit growth mismatches observed M2, don't claim macro calibration succeeded.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from economy_v81 import Economy,Config,MODES,POLICIES,ROOT
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)

def fiscal_stat(r,start,end):
    x=r.iloc[start-1]; y=r.iloc[end-1]; return (y.public_spend-x.public_spend)/max(y.taxes_total-x.taxes_total,1.e-10)

def fit(seeds=(7000,7001)):
    b=pd.read_csv(ROOT/'data/federal_budget_observed_2019_2025.csv').set_index('year')
    obs=(b.loc[2019:2022].expenditure_billion_rub/b.loc[2019:2022].revenue_billion_rub).mean()
    m=pd.read_csv(ROOT/'data/m2_secondary_2016_2025.csv').set_index('year')
    observed_growth=float(m.loc[2022,'m2_trillion_rub']/m.loc[2019,'m2_trillion_rub'])
    results=[]
    for fiscal_scale in [5.,6.5,7.5,8.5,10.]:
        for credit_scale in [0.5,1.,2.]:
            for seed in seeds:
                r=Economy(Config('russia_mixed','baseline',seed,False,.4,fiscal_scale,credit_scale)).run(84)
                ratio=fiscal_stat(r,36,84)
                deposits=r.iloc[83].private_deposits/r.iloc[47].private_deposits
                results.append(dict(seed=seed,fiscal_scale=fiscal_scale,credit_scale=credit_scale,observed_fiscal_ratio=obs,model_fiscal_ratio=ratio,observed_M2_factor_2019_2022=observed_growth,model_deposit_factor=deposits,bank_resolutions=r.iloc[83].bank_resolutions,bank_capratio=r.iloc[83].bank_capratio))
    grid=pd.DataFrame(results);grid.to_csv(OUT/'calibration_train_grid.csv',index=False)
    means=grid.groupby(['fiscal_scale','credit_scale']).mean(numeric_only=True).reset_index()
    means['fiscal_err']=abs(means.model_fiscal_ratio-obs)
    means['m2_err']=abs(np.log(means.model_deposit_factor/observed_growth))
    means['total_error']=means.fiscal_err+means.m2_err # EQUAL scoring weights; transparent and arbitrary, test robustness
    means=means.sort_values('total_error'); means.to_csv(OUT/'calibration_objective_grid.csv',index=False)
    best=means.iloc[0]
    pick=dict(fiscal_scale=float(best.fiscal_scale),credit_scale=float(best.credit_scale),observed_fiscal_ratio=float(obs),observed_M2_factor=float(observed_growth),model_fiscal_ratio=float(best.model_fiscal_ratio),model_deposit_factor=float(best.model_deposit_factor),deposit_growth_not_adequately_calibrated=bool(abs(np.log(best.model_deposit_factor/observed_growth))>.15))
    (OUT/'train_only_frozen_calibration.json').write_text(json.dumps(pick,ensure_ascii=False,indent=2),encoding='utf8')
    return pick

def run(nseeds=3):
    p=fit(); print('TRAIN FROZEN PARAMETERS',p,flush=True)
    summaries=[];paths=[]
    for seed in range(9800,9800+nseeds):
        for stress in [False,True]:
            for mode in MODES:
                for policy in POLICIES:
                    r=Economy(Config(mode,policy,seed,stress,.4,p['fiscal_scale'],p['credit_scale'])).run(120)
                    old=r.iloc[83];z=r.iloc[119]
                    summaries.append(dict(seed=seed,stress=stress,mode=mode,policy=policy,oos_sales=z.real_final_sales-old.real_final_sales,oos_consumption=z.real_consumption-old.real_consumption,oos_real_output=z.real_output_gross-old.real_output_gross,bank_resolutions=z.bank_resolutions,bank_capital_ratio=z.bank_capratio,gini=z.hh_liquid_gini,escrow_issued=z.escrow_issued,escrow_matched=z.bank_matched,escrow_unlocked=z.escrow_unlocked,demand_vouchers=z.demand_vouchers,financial_residual=z.financial_residual,input_output_residual=z.input_output_residual,deposit_factor_full_period=z.private_deposits/r.iloc[0].private_deposits,public_spending_tax_ratio=fiscal_stat(r,84,120)))
                    if seed==9800 and not stress and policy in ['baseline','escrow_demand']:paths.append(r.assign(target_mode=mode,target_policy=policy))
        print('DONE OOS SEED',seed,flush=True)
    df=pd.DataFrame(summaries);df.to_csv(OUT/'paired_four_economies_oos.csv',index=False)
    if paths:pd.concat(paths,ignore_index=True).to_csv(OUT/'representative_paths.csv',index=False)
    means=df.groupby(['stress','mode','policy']).agg(n=('seed','size'),mean_sales=('oos_sales','mean'),mean_consumption=('oos_consumption','mean'),mean_banks=('bank_resolutions','mean'),gini=('gini','mean'),escrow_unlocked=('escrow_unlocked','mean'),demand=('demand_vouchers','mean'),bank_capital=('bank_capital_ratio','mean')).reset_index()
    means.to_csv(OUT/'group_means.csv',index=False)
    for st in [False,True]:
        print('STRESS',st,means[(means.stress==st)&(means.policy.isin(['baseline','escrow_demand']))][['mode','policy','mean_sales','mean_consumption','mean_banks']].to_string(index=False),flush=True)
    checks={'n_runs':len(df),'max_financial_error':float(df.financial_residual.max()),'max_real_IO_error':float(df.input_output_residual.max()),'max_escrow_match_error':float(np.max(abs(df.escrow_unlocked-df.escrow_matched))),'max_seeds':nseeds,'historical_training_only':True,'missing_actual_ROSSTAT_IO_matrix':True,'calibrated_banking_M2_rejected':p['deposit_growth_not_adequately_calibrated']}
    assert checks['max_financial_error']<1e-5 and checks['max_real_IO_error']<1e-6 and checks['max_escrow_match_error']<1e-8
    (OUT/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8')
    print('CHECKS',checks,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,default=3);run(ap.parse_args().seeds)