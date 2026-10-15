"""TRAIN 2016–22 / FREEZE / OOS 2023–25. Paired seeds and policy ablations.
Training objective contains only past simulation observables. GDP/CPI observations
from the OOS period are NOT accessed by code; they appear in the final report only.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from economy_v8 import Economy,Config,MODES,POLICIES,ROOT,SECTORS,A
OUT=ROOT/'results';OUT.mkdir(parents=True,exist_ok=True)

def score(df):
    r=df.iloc[-1];return float(r.real_consumption-20*r.bank_resolutions-.003*r.input_output_residual)

def tune(train_seeds=4):
    rows=[];choice={}
    for mode in MODES:
        for gain in [.25,.50,.75]:
            vals=[]
            for i in range(train_seeds):
                z=Economy(Config(mode,'baseline',7000+i,False,gain)).run(84)
                vals.append(score(z))
            rows.append(dict(mode=mode,response=gain,train_score_mean=np.mean(vals),train_score_sd=np.std(vals)))
        s=pd.DataFrame(rows);ss=s[s['mode']==mode].sort_values(['train_score_mean','response'],ascending=[False,True])
        choice[mode]=float(ss.iloc[0].response)
    pd.DataFrame(rows).to_csv(OUT/'train_2016_2022_grid.csv',index=False)
    (OUT/'frozen_policies.json').write_text(json.dumps(choice,ensure_ascii=False,indent=2),encoding='utf8')
    return choice

def run(nseeds=6):
    frozen=tune(4)
    summary=[];paths=[];annual=[]
    for si in range(nseeds):
        for stress in (False,True):
            for mode in MODES:
                for policy in POLICIES:
                    seed=9800+si
                    df=Economy(Config(mode,policy,seed,stress,frozen[mode])).run(120)
                    at84=df.iloc[83];last=df.iloc[119]
                    row={k:last[k] for k in ('mode','policy','seed','stress','bank_resolutions','firm_defaults','hh_liquid_gini','bank_capratio','gov_debt','debt_to_bank_deposits','escrow_issued','escrow_unlocked','bank_matched','locked_escrow','demand_vouchers','financial_residual','input_output_residual')}
                    for k in ('real_final_sales','real_consumption','real_output_gross','final_supply','intermediates'):
                        row['oos_'+k]=last[k]-at84[k]
                    row['oos_annualized_inflation']=100*(np.log(last.price_index/at84.price_index))/3
                    row['stress_label']='crisis' if stress else 'normal'
                    summary.append(row)
                    if si==0:
                        paths.append(df)
                    for yr in range(2017,2026):
                        offset=(yr-2016)*12
                        prev=df.iloc[offset-13];end=df.iloc[offset-1]
                        oldprev=df.iloc[offset-25] if offset>=24 else None
                        if oldprev is not None:
                            gdp_previous=prev.real_final_sales-oldprev.real_final_sales
                            gdp_current=end.real_final_sales-prev.real_final_sales
                            gdp_growth=100*(gdp_current/max(gdp_previous,1e-6)-1)
                        else:gdp_growth=np.nan
                        price_change=100*(end.price_index/prev.price_index-1)
                        annual.append(dict(mode=mode,policy=policy,stress=int(stress),seed=seed,year=yr,split='OOS' if yr>=2023 else 'TRAIN',model_real_final_sales_yoy_percent=gdp_growth,model_cpi_yoy_percent=price_change))
        print('evaluated seeds',si+1,'/',nseeds,flush=True)
    r=pd.DataFrame(summary); r.to_csv(OUT/'v8_results.csv',index=False)
    pd.concat(paths,ignore_index=True).to_csv(OUT/'representative_paths.csv',index=False)
    a=pd.DataFrame(annual);a.to_csv(OUT/'annual_model_predictions.csv',index=False)
    ob=pd.read_csv(ROOT/'data/rus_macro_2016_2025_observed.csv')
    # Out-of-sample observations attached AFTER freezing policies and completing simulations.
    oos=a[a.year>=2023].merge(ob[['year','real_gdp_yoy_percent','cpi_dec_to_dec_percent']],on='year',how='left')
    oos['abs_gdp_error_pp']=abs(oos.model_real_final_sales_yoy_percent-oos.real_gdp_yoy_percent)
    oos['abs_inflation_error_pp']=abs(oos.model_cpi_yoy_percent-oos.cpi_dec_to_dec_percent)
    oos.to_csv(OUT/'oos_2023_2025_predictions_vs_observed.csv',index=False)
    means=r.groupby(['stress_label','policy','mode']).agg(n=('seed','count'),real_final_sales=('oos_real_final_sales','mean'),real_consumption=('oos_real_consumption','mean'),bank_resolutions=('bank_resolutions','mean'),gini=('hh_liquid_gini','mean'),bank_capratio=('bank_capratio','mean'),escrow_unlocked=('escrow_unlocked','mean'),demand_vouchers=('demand_vouchers','mean'))
    means.to_csv(OUT/'group_means.csv')
    paired=[]
    for (seed,stress,policy),z in r.groupby(['seed','stress_label','policy']):
        b=z.set_index('mode').loc['capitalism']
        for mode in MODES:
            t=z.set_index('mode').loc[mode]
            paired.append(dict(seed=seed,stress=stress,policy=policy,mode=mode,delta_final_sales=t.oos_real_final_sales-b.oos_real_final_sales,delta_consumption=t.oos_real_consumption-b.oos_real_consumption,delta_bank_resolutions=t.bank_resolutions-b.bank_resolutions))
    pd.DataFrame(paired).to_csv(OUT/'paired_vs_capitalism.csv',index=False)
    print(means.to_string())
    print('MAX financial',r.financial_residual.max(),'MAX physical',r.input_output_residual.max(),'n=',len(r))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=6)
    run(p.parse_args().seeds)