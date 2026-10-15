import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from macro_v91 import MacroTwin,Config
from twin import MODES
class TestMacroV91(unittest.TestCase):
 def setUp(self):
  self.m=MacroTwin(Config(mode='synergy',policy='escrow_demand',seed=123,stress=True,nmonths=16))
 def test_all_same_goods_technologies(self):
  for mode in MODES:
   a=MacroTwin(Config(mode=mode,seed=1));b=MacroTwin(Config(mode=MODES[0],seed=1))
   np.testing.assert_allclose(a.capacity,b.capacity)
 def test_real_gva_not_same_as_gross(self):
  r=self.m.run();self.assertGreater(r[-1]['gva_month_real_proxy'],0)
  self.assertLess(r[-1]['gva_month_real_proxy'],r[-1]['physical_output'])
 def test_cpi_and_inflation_measured(self):
  r=self.m.run();self.assertTrue(np.isfinite(r[-1]['cpi_yoy_pct']))
  self.assertGreater(r[-1]['cpi_fixed_basket'],0)
 def test_shares_conserved(self):
  self.m.run();np.testing.assert_allclose(self.m.equity_shares.sum(axis=1),self.m.shares_issued,atol=1e-7)
 def test_trades_settled_with_real_balances(self):
  self.m.run();self.assertGreater(self.m.equity_cum_volume,0)
  self.m._audit()
 def test_dividends_not_created_out_of_air(self):
  self.m.run();self.assertGreaterEqual(self.m.dividends_paid,0)
  self.m._audit()
 def test_external_reserves_only_after_exports(self):
  self.m.run();self.assertAlmostEqual(self.m.row_reserves.sum(),self.m.world_export_receipts.sum()-self.m.world_import_payments.sum(),places=6)
 def test_imported_goods_audited(self):
  self.m.run();self.assertLess(self.m.history[-1]['physical_audit_error'],1e-7)
 def test_bank_balances(self):
  self.m.run();self.assertLess(self.m.history[-1]['bank_audit_error'],1e-7)
 def test_household_wealth_inequality(self):
  self.m.run();r=self.m.history[-1];self.assertTrue(0<=r['household_wealth_gini_with_equity']<=1)
 def test_capital_mark_is_not_money_supply(self):
  self.m.run();r=self.m.history[-1];self.assertNotAlmostEqual(r['equity_market_cap_mark'],r['m2_proxy'])
 def test_nonnegative_reserves_and_cash(self):
  self.m.run();self.assertTrue((self.m.row_reserves>=-1e-10).all());self.assertTrue((self.m.hbalance>=-1e-10).all())
if __name__=='__main__':unittest.main()