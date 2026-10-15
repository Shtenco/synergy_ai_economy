import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from economy_v82 import Economy82,Config82
class TestMonetaryOperations(unittest.TestCase):
 def instance(self):return Economy82(Config82(seed=17,household_credit_rate=.003))
 def test_balance_starts(self):
  e=self.instance();self.assertLess(e.maxerr,1e-7)
 def test_cash_conversion_preserves_m2_and_equity(self):
  e=self.instance();e0=e.equity().copy();m2=e.money_supply_proxy();e.convert_cash(np.ones(400)*.15)
  self.assertAlmostEqual(e.money_supply_proxy(),m2,places=8)
  np.testing.assert_allclose(e.equity(),e0,atol=1e-8)
  self.assertLess(e.maxerr,1e-7)
 def test_cash_redeposit_preserves_m2(self):
  e=self.instance();m2=e.money_supply_proxy();e.deposit_cash(e.cash_household*.09)
  self.assertAlmostEqual(e.money_supply_proxy(),m2,places=8)
 def test_household_loan_creates_deposit(self):
  e=self.instance();old=e.hl.sum();dep=e.d_by_bank().sum();e.household_credit_cycle(0)
  self.assertGreater(e.hl.sum(),old)
  self.assertAlmostEqual(e.d_by_bank().sum()-dep,e.credit_h_issued-e.credit_h_repaid-e.credit_h_interest,places=7)
  self.assertLess(e.maxerr,1e-7)
 def test_loan_principal_repayment_destroys_money(self):
  e=self.instance();e.c.household_credit_rate=0
  before_m=e.money_supply_proxy();before_loans=e.hl.sum()
  e.household_credit_cycle(0)
  self.assertLess(e.hl.sum(),before_loans)
  self.assertLess(e.money_supply_proxy(),before_m)
 def test_bank_capital_guard(self):
  e=self.instance();e.c.min_capital_ratio=9.0
  e.c.household_credit_rate=.5
  before=e.hl.sum();e.household_credit_cycle(0)
  self.assertLess(e.hl.sum(),before+1e-10)
 def test_longrun_and_escrow(self):
  e=Economy82(Config82(mode='synergy_cyber',policy='escrow_demand',seed=101,stress=True,household_credit_rate=0))
  d=e.run(120)
  self.assertLess(e.maxerr,1e-7)
  self.assertLess(e.ioresmax,1e-7)
  self.assertAlmostEqual(e.released,e.matched,places=9)
  self.assertTrue(np.all(d.m2_proxy>0))
 def test_cash_is_in_M2_proxy(self):
  e=self.instance()
  self.assertAlmostEqual(e.money_supply_proxy(),e.d_by_bank().sum()+e.cbr_currency,places=8)
if __name__=='__main__': unittest.main()