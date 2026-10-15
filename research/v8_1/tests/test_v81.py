import unittest, sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from economy_v81 import Economy,Config,A,MODES
class V81Tests(unittest.TestCase):
 def test_budget_sources_have_seven_years(self):
  b=pd.read_csv(ROOT/'data/federal_budget_observed_2019_2025.csv')
  self.assertEqual(b.year.tolist(),list(range(2019,2026)))
  self.assertTrue((b.expenditure_billion_rub>0).all())
 def test_no_io_claim(self):
  self.assertTrue((A>=0).all());self.assertLess(float(A.sum(axis=0).max()),1)
 def test_m2_train_series_dates(self):
  z=pd.read_csv(ROOT/'data/m2_secondary_2016_2025.csv')
  self.assertEqual(z.year.tolist(),list(range(2016,2026)))
 def test_escrow_ledger(self):
  z=Economy(Config('synergy_cyber','escrow_demand',7,True,.4,7.5,2.0)).run(20)
  self.assertAlmostEqual(float(z.iloc[-1].escrow_unlocked),float(z.iloc[-1].bank_matched))
  self.assertLess(float(z.financial_residual.max()),1e-5)
 def test_bank_finance_conserves(self):
  for m in MODES:
   z=Economy(Config(m,'baseline',7,False,.4,7.5,2.0)).run(24)
   self.assertLess(float(z.financial_residual.max()),1e-5)
   self.assertLess(float(z.input_output_residual.max()),1e-6)
 def test_no_future_fiscal_data_in_fit(self):
  s=(ROOT/'src/calibrate_v81.py').read_text()
  self.assertIn('loc[2019:2022]',s)
  self.assertNotIn("loc[2023:2025]",s)
if __name__=='__main__': unittest.main()