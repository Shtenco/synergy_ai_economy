import pandas as pd,numpy as np,pathlib,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=pathlib.Path(__file__).resolve().parents[1]
fig=p/'figures';fig.mkdir(exist_ok=True)
data={}
for mode in ['capitalism','soviet','russia_mixed','synergy']:
 z=pd.read_csv(p/'results'/f'paths_{mode}_stress1.csv')
 data[mode]=z[(z.policy=='escrow_demand')&(z.seed==2000)]
for col,title,name in [
 ('gva_annual_real_proxy','Annual real value-added [experimental fixed-price units]','09_gva_timepath.png'),
 ('cpi_fixed_basket','Consumer price index from synthetic market prices','10_cpi_timepath.png'),
 ('median_real_household_deposits','Median real household deposits [synthetic units]','11_family_liquid_wealth.png'),
 ('row_external_reserves_asset','External ROW asset claims from real simulated exports','12_external_reserves.png'),
 ('equity_price_index_100','Secondary-share market index, start=100 [not real MOEX]','13_equity_index.png')]:
 f,a=plt.subplots(figsize=(9,4.6))
 for mode,z in data.items():
  if col=='equity_price_index_100' and mode=='soviet':continue
  a.plot(2016+(z.month.values-1)/12,z[col].values,label=mode.replace('_',' '))
 a.axvspan(2023,2026,color='gray',alpha=.1)
 a.legend(loc='best');a.grid(alpha=.2);a.set_title(title);a.set_xlabel('Model year, 2016–2025');f.tight_layout();f.savefig(fig/name,dpi=140);plt.close(f)
print('created 5 extra figures')