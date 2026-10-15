"""Scenario outcomes, comparisons, independent official benchmarks, and visual evidence."""
import pathlib,glob,hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=pathlib.Path(__file__).resolve().parents[1]
rd=P/'results'; fns=list(rd.glob('runs_*_stress*.csv'))
runs=pd.concat([pd.read_csv(x) for x in fns],ignore_index=True)
runs.to_csv(rd/'all_96_runs.csv',index=False)
metrics=['oos_gva_real','oos_real_gva_growth_last_year_pct','oos_cpi_inflation_yoy_pct',
 'oos_price_instability_ann_pct','oos_real_median_liquid_wealth',
 'oos_real_consumption_per_household_last_year','oos_real_total_household_marked_wealth',
 'oos_household_wealth_gini','oos_equity_market_index','oos_equity_traded_cash',
 'oos_row_external_reserves','oos_row_reserve_increment','oos_row_exports_cash','oos_row_imports_cash',
 'oos_domestic_supply_pct','oos_bank_capital_ratio','oos_gov_debt_gva_pct',
 'oos_project_commissioned','oos_project_writeoffs']
groups=runs.groupby(['stress','mode','policy'])[metrics].agg(['mean','std','count'])
groups.columns=['_'.join(z) for z in groups.columns];groups.reset_index().to_csv(rd/'multi_kpi_group_stats.csv',index=False)
# Matched seeds for all policy cases; no unmatched averages.
wide=runs.pivot_table(index=['stress','mode','seed'],columns='policy',values=metrics)
deltas=[]
for idx in wide.index:
 for m in metrics:
  if (m,'baseline') in wide.columns and (m,'escrow_demand') in wide.columns:
   vals=wide.loc[idx];deltas.append(dict(stress=idx[0],mode=idx[1],seed=idx[2],metric=m,delta_escrow_demand_minus_baseline=vals[(m,'escrow_demand')]-vals[(m,'baseline')]))
pd.DataFrame(deltas).to_csv(rd/'paired_policy_deltas.csv',index=False)
# Compare Russian model annual percentage growth and CPI to real observed annual data (not fitted).
obs=pd.read_csv(P/'data/russia_observed_2016_2025.csv')
annual=[]
for s in (0,1):
 p=rd/f'paths_russia_mixed_stress{s}.csv'
 if p.is_file():
  z=pd.read_csv(p)
  z=z[(z.policy=='baseline')&(z.seed==2000)]
  for _,row in z[(z.month%12==0)&(z.month>=24)].iterrows():
   y=2015+int(row.month/12)
   d=obs.loc[obs.year==y]
   if len(d):
    d=d.iloc[0]
    annual.append(dict(stress=s,year=y,model_gva_yoy_pct=row.real_gva_yoy_pct,observed_rus_gdp_yoy_pct=d.real_gdp_yoy_percent,model_cpi_yoy_pct=row.cpi_yoy_pct,observed_rus_cpi_yoy_pct=d.cpi_dec_to_dec_percent))
annual=pd.DataFrame(annual);annual.to_csv(rd/'russia_official_gdp_cpi_oos_comparison.csv',index=False)
errs=[]
for st,g in annual[annual.year>=2023].groupby('stress'):
 errs.append(dict(stress=int(st),year_range='2023-2025',gva_growth_mae_pp=float(np.mean(abs(g.model_gva_yoy_pct-g.observed_rus_gdp_yoy_pct))),cpi_inflation_mae_pp=float(np.mean(abs(g.model_cpi_yoy_pct-g.observed_rus_cpi_yoy_pct))),status='FAILED macro-validation (scenario-conditioned historical shocks, no independent real-economy forecasts)'))
pd.DataFrame(errs).to_csv(rd/'macro_validation_errors.csv',index=False)
crisis=runs[(runs.stress==1)&(runs.policy=='escrow_demand')].groupby('mode')[metrics].mean().reset_index()
crisis.to_csv(rd/'crisis_escrow_demand_macro_dashboard.csv',index=False)
# Dashboard plots
fig=P/'figures';fig.mkdir(exist_ok=True)
plots=[('oos_gva_real','01_real_value_added.png','Real value added, OOS 2023-2025 [synthetic units]'),
 ('oos_real_gva_growth_last_year_pct','02_real_growth.png','Real value added growth, 2025 [%]'),
 ('oos_cpi_inflation_yoy_pct','03_cpi.png','Synthetic CPI change, 2025 [%]'),
 ('oos_real_median_liquid_wealth','04_household_wealth.png','Median real household deposits, 2025 [synthetic units]'),
 ('oos_equity_market_index','05_stock_market.png','Synthetic traded equity price index, start=100'),
 ('oos_row_external_reserves','06_foreign_reserves.png','ROW reserve claims, 2025 [synthetic units]'),
 ('oos_domestic_supply_pct','07_local_supply.png','Domestic gross production share proxy [%]'),
 ('oos_bank_capital_ratio','08_bank_stability.png','Minimum bank capital ratio [fraction]')]
for key,fn,title in plots:
 fig1,ax=plt.subplots(figsize=(9,4.6))
 x=np.arange(4);labels=['Capitalism','USSR-like','Russia mixed','Synergy']
 lookup=['capitalism','soviet','russia_mixed','synergy']
 for i,policy in enumerate(('baseline','escrow_demand')):
  values=[runs[(runs.stress==1)&(runs.mode==mode)&(runs.policy==policy)][key].mean() for mode in lookup]
  ax.bar(x+(i-.5)*.35,values,.34,label=policy)
 ax.set_xticks(x,labels);ax.set_title(title);ax.legend();ax.grid(axis='y',alpha=.20);fig1.tight_layout();fig1.savefig(fig/fn,dpi=140);plt.close(fig1)
checks={'scenario_count':len(runs),'scenario_groups':len(groups),'max_goods_error':float(runs.max_goods_error.max()),'max_bank_error':float(runs.max_bank_error.max()),'historical_model_cpi_gdp_validation':'FAILED / NOT CALIBRATED','stock_prices':'SYNTHETIC settled bids/offers, not MOEX','reserve_assets':'SYNTHETIC ROW export proceeds, not actual CBR international reserves','real_gva':'production-account approximation, not national accounts GDP','household_wealth':'deposit+equity marks; excludes housing, liabilities and pension claims','capital_controls':'fixed rule, not actual central bank law','physical_coefficients':'ASSUMPTIONS; no Rosstat IO XLSX was loaded'}
(rd/'verification_v91.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
print('All runs:',len(runs),'historical OOS error:',errs)
print(crisis[['mode','oos_gva_real','oos_real_gva_growth_last_year_pct','oos_cpi_inflation_yoy_pct','oos_real_median_liquid_wealth','oos_equity_market_index','oos_row_external_reserves']].round(2).to_string(index=False))