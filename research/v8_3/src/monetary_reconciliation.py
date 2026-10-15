"""Counterpart reconciliation of V8.3 M2 proxy, without forced emissions.
Under simplified domestic bank and central-bank accounting:
M2_proxy = (loans_H+loans_F) + (gov bonds bank + gov bonds CB)
            - bank_equity - CB_equity - treasury_deposit.
This is a *model identity*, not a full empirical counterparts table from CBR.
"""
from pathlib import Path
import numpy as np,pandas as pd,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from economy_v83 import Economy83,Config83
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'

def get_sector_claims(e,month):
    # Cash is an explicit CB liability, reserves are CB liabilities to banks.
    cb_eq=e.cb_bonds+e.refi.sum()-e.reserve.sum()-e.treasury-e.cbr_currency
    loans=e.l_by_bank().sum()
    gov_bonds=e.bank_bonds.sum()+e.cb_bonds
    bank_eq=e.equity().sum()
    treasury=e.treasury
    m2=float(e.money_supply_proxy())
    res=m2-(loans+gov_bonds-bank_eq-cb_eq-treasury)
    assert abs(res)<1e-7,res
    return dict(month=month,loans=loans,government_bonds=gov_bonds,
                bank_equity=bank_eq,central_bank_equity=cb_eq,
                treasury_deposit=treasury,m2_proxy=m2,
                deposits=e.d_by_bank().sum(),currency_outside=e.cbr_currency,
                residual=res,external_NFA_modelled=0.0)

def run():
    e=Economy83(Config83(mode='russia_mixed',policy='baseline',seed=9800,
      credit_scale=2.,fiscal_scale=7.5,household_credit_rate=0.,w_shortage=.75))
    rows=[]
    for month in range(120):
        e.month_step(month);rows.append(get_sector_claims(e,month+1))
    r=pd.DataFrame(rows);r.to_csv(OUT/'monetary_counterparts_120_months.csv',index=False)
    bas=r.loc[47];end=r.loc[83]
    decom={
      'change_loans':end.loans-bas.loans,
      'change_government_bonds':end.government_bonds-bas.government_bonds,
      'negative_change_bank_equity':-(end.bank_equity-bas.bank_equity),
      'negative_change_cb_equity':-(end.central_bank_equity-bas.central_bank_equity),
      'negative_change_treasury':-(end.treasury_deposit-bas.treasury_deposit)}
    m2_change=end.m2_proxy-bas.m2_proxy
    obs=pd.read_csv(ROOT/'data/m2_secondary_2016_2025.csv').set_index('year')
    target=float(obs.loc[2022,'m2_trillion_rub']/obs.loc[2019,'m2_trillion_rub'])
    result={'period':'2019-12 to 2022-12','model_m2_start':float(bas.m2_proxy),
        'model_m2_end':float(end.m2_proxy),'model_factor':float(end.m2_proxy/bas.m2_proxy),
        'secondary_observed_m2_factor':target,'model_m2_delta':float(m2_change),
        'counterparts':decom,'decomposition_residual':float(sum(decom.values())-m2_change),
        'empirical_validation_pass':False,'critical_missing_channels':[
        'nonfinancial resident money-holder coverage','net foreign assets and forex settlement',
        'banks other assets/liabilities','CBR actual annual monetary aggregates',
        'bank nonperforming loan classification','income-expenditure-consistent nominal GDP scale'],
        'warning':'No residual credit injection or fiat emission has been applied to force M2 matching.'}
    (OUT/'monetary_gap_explained.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':run()