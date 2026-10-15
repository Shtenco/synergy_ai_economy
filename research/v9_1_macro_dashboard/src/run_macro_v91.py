"""Use V9's frozen 2016-2022 policy weights; no re-training on 2023-2025.
Run independent, scenario-matched Monte Carlo across all regimes/policies.
"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from macro_v91 import MacroTwin,Config
from twin import MODES,POLICIES
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--stress',type=int,choices=[0,1],required=True);parser.add_argument('--seeds',type=int,default=4);parser.add_argument('--mode',choices=MODES,required=True);args=parser.parse_args()
prev=Path(__file__).resolve().parents[2]/'V9_1_MACRO/base/SINERGY_ECONOMIC_DIGITAL_TWIN_V9/results/frozen_policies.json'
# Standalone copy of V9 frozen parameters, committed with artifact.
weights=json.loads((root/'data/frozen_v9_train_only_weights.json').read_text())['best_policy_weight']
rows=[];paths=[]
for mode in [args.mode]:
 for policy in POLICIES:
  for seed in range(2000,2000+args.seeds):
   e=MacroTwin(Config(mode=mode,policy=policy,seed=seed,stress=bool(args.stress),planning_weight=weights[mode],nmonths=120))
   hist=e.run();last=hist[-1];past=hist[83]
   r=dict(mode=mode,policy=policy,stress=args.stress,seed=seed,planning_weight=weights[mode],
     oos_gva_real=sum(z['gva_month_real_proxy'] for z in hist[84:]),
     oos_gva_nominal=sum(z['gva_month_nominal_proxy'] for z in hist[84:]),
     oos_household_real_consumption=(last['consumption']-past['consumption']),
     oos_total_physical_output=last['physical_output']-past['physical_output'],
     oos_real_gva_growth_last_year_pct=last['real_gva_yoy_pct'],
     oos_nominal_gva_growth_last_year_pct=last['nominal_gva_yoy_pct'],
     oos_cpi_inflation_yoy_pct=last['cpi_yoy_pct'],
     oos_price_instability_ann_pct=last['inflation_instability_ann_pct'],
     oos_real_median_liquid_wealth=last['median_real_household_deposits'],
     oos_real_total_household_marked_wealth=last['household_total_wealth_real_mark'],
     oos_real_consumption_per_household_last_year=last['annual_real_consumption_per_hh'],
     oos_household_wealth_gini=last['household_wealth_gini_with_equity'],
     oos_equity_market_index=last['equity_price_index_100'],
     oos_equity_traded_cash=last['equity_cumulative_cash_turnover']-past['equity_cumulative_cash_turnover'],
     oos_equity_dividends=last['equity_cumulative_cash_dividends']-past['equity_cumulative_cash_dividends'],
     oos_row_external_reserves=last['row_external_reserves_asset'],
     oos_row_reserve_increment=last['row_external_reserves_asset']-past['row_external_reserves_asset'],
     oos_row_exports_cash=last['row_export_receipts']-past['row_export_receipts'],
     oos_row_imports_cash=last['row_import_payments']-past['row_import_payments'],
     oos_domestic_supply_pct=last['domestic_gross_supply_share_proxy_pct'],
     oos_m2_proxy=last['m2_proxy'],
     oos_bank_capital_ratio=last['bank_min_capital_ratio'],
     oos_gov_debt_gva_pct=last['government_debt_to_annual_gva_proxy_pct'],
     oos_project_commissioned=last['commissioned']-past['commissioned'],
     oos_project_writeoffs=last['capex_written_off']-past['capex_written_off'],
     max_goods_error=max(z['physical_audit_error'] for z in hist),
     max_bank_error=max(z['bank_audit_error'] for z in hist))
   rows.append(r)
   if seed==2000:paths += hist
   assert np.isclose(e.row_reserves.sum(),e.world_export_receipts.sum()-e.world_import_payments.sum(),rtol=0,atol=1e-7)
   assert np.allclose(e.equity_shares.sum(axis=1),e.shares_issued,atol=1e-7)
   print('COMPLETED',mode,policy,'stress',args.stress,'seed',seed,flush=True)
out=root/'results';out.mkdir(exist_ok=True)
for n,data in [('runs_'+args.mode+'_stress'+str(args.stress)+'.csv',rows),('paths_'+args.mode+'_stress'+str(args.stress)+'.csv',paths)]:
 with (out/n).open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,data[0]);w.writeheader();w.writerows(data)
print('TOTAL',len(rows),'STRESS',args.stress)