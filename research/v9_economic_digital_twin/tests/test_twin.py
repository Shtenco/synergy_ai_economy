import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from twin import Twin,Config,ARCHETYPES,MODES,POLICIES,A,N,S

class PhysicalAndMonetaryTests(unittest.TestCase):
 def test_5_by_8_and_24(self):
  a=Twin(Config())
  self.assertEqual(a.stock.shape,(5,8));self.assertEqual(len(ARCHETYPES),24)
  self.assertTrue((np.linalg.eigvals(A).real<1).all())
 def test_120_months_full_stock_and_bank_audit(self):
  for mode in MODES:
   e=Twin(Config(mode=mode,policy='escrow_demand',nmonths=120,seed=41,stress=True))
   e.run();e._audit()
   self.assertTrue(np.all(e.stock>=-1e-7))
   self.assertLess(e.history[-1]['physical_audit_error'],1e-7)
   self.assertLess(e.history[-1]['bank_audit_error'],1e-7)
 def test_negative_commodity_detected(self):
  e=Twin(Config())
  e.stock[0,3]-=.01
  with self.assertRaises(AssertionError):e._audit()
 def test_tampered_bank_claim_detected(self):
  e=Twin(Config())
  e.bank_equity_claim[0]+=1
  with self.assertRaises(AssertionError):e._audit()
 def test_offbalance_escrow_no_created_money(self):
  e=Twin(Config(policy='escrow',nmonths=30))
  before=(e.hdep.sum()+e.fdep.sum(),e.breserve.sum(),e.loan.sum())
  e.t=12;e._offer_escrow()
  after=(e.hdep.sum()+e.fdep.sum(),e.breserve.sum(),e.loan.sum())
  self.assertEqual(before,after)
  self.assertGreater(e.escrow_locked.sum(),0)
  e._audit()
 def test_match_is_real_equity_and_second_loan(self):
  e=Twin(Config(policy='escrow',seed=3));e.t=12;e._offer_escrow()
  cb=e.bank_refi[0];f=e.fdep[0,2];lo=e.loan[0,2];stake=e.bank_equity_claim[0];r=e.breserve[0]
  total=e._finance(0,2,4.0)
  self.assertAlmostEqual(total,8)
  self.assertAlmostEqual(e.fdep[0,2]-f,8)
  self.assertAlmostEqual(e.loan[0,2]-lo,4)
  self.assertAlmostEqual(e.bank_equity_claim[0]-stake,4)
  self.assertAlmostEqual(e.bank_refi[0]-cb,4)
  self.assertAlmostEqual(e.breserve[0]-r,4)
  self.assertAlmostEqual(e.unlocked,e.matched)
  e._audit()
 def test_no_leak_on_aborted_first_loss(self):
  e=Twin(Config(policy='escrow'))
  e.t=12;e._offer_escrow();e._finance(0,2,4)
  e._new_project(0,2,2,8,bank_equity_at_risk=4)
  e.projects[0].funded=2;e.capital_cost=2
  old=e.stock[0].copy();e.total_used+=old;e.stock[0]=0
  e.projects[0].launched=-30;e._procure();e._audit()
  self.assertAlmostEqual(e.capital_abandoned,2)
  self.assertLess(e.bank_equity_claim[0],4)
 def test_crossborder_goods_cash_and_cb(self):
  e=Twin(Config())
  amount=3.5
  before=e.stock.sum(axis=0).copy();cb=e.cb_external.copy()
  moved=e._transfer((0,2),(1,2),amount)
  self.assertAlmostEqual(moved*e.price[0,2],amount)
  np.testing.assert_allclose(e.stock.sum(axis=0),before)
  self.assertAlmostEqual(e.cb_external.sum(),cb.sum())
  self.assertAlmostEqual(e.fdep[0,2]-220,amount)
  self.assertAlmostEqual(e.fdep[1,2]-220,-amount)
  e._audit()
 def test_project_has_no_instant_capacity(self):
  e=Twin(Config());k=e.capacity.copy()
  self.assertTrue(e._new_project(1,3,10,10))
  np.testing.assert_array_equal(e.capacity,k)
  e._audit()
 def test_real_inputs_required_for_commission(self):
  e=Twin(Config());self.assertTrue(e._new_project(0,3,10,10))
  e.stock[0,:]=0.;e.total_used+=e.initial_physical/N
  # Physical stocks adjusted manually only for test setup; project must not advance.
  e._procure();self.assertEqual(e.project_completed,0)
 def test_no_effect_without_program(self):
  a=Twin(Config(mode='capitalism',policy='baseline',seed=2,nmonths=6))
  a.run();self.assertEqual(a.unlocked,0);self.assertEqual(a.issued,0)
 def test_fiscal_is_not_free_private_asset(self):
  e=Twin(Config(policy='demand'))
  old=(e.govdebt.copy(),e.hdep.copy())
  e.t=14;e.short[:]=.5;e._voucher()
  self.assertGreater(e.fiscal,0)
  np.testing.assert_allclose(e.govdebt-old[0],e.hdep-old[1])
  e._audit()
 def test_energy_grid_bottleneck(self):
  x=Twin(Config())
  x.grid[:]=0;x.stock[:,1]=0
  x.total_used[1]+=x.initial_physical[1]
  gross=x._produce()
  self.assertTrue(np.all(gross[:,1]==0))
  self.assertLess(gross.sum(),100)
 def test_same_initial_physical_technologies(self):
  a=Twin(Config(mode='capitalism',seed=8))
  for m in MODES:
   b=Twin(Config(mode=m,seed=8))
   np.testing.assert_array_equal(b.stock,a.stock)
   np.testing.assert_array_equal(b.capacity,a.capacity)
 def test_fair_information_budget(self):
  for m in MODES:
   a=Twin(Config(mode=m,inspection_budget=5))
   self.assertEqual(a.c.inspection_budget,5)
 def test_no_future_fx(self):
  x=Twin(Config(seed=1,nmonths=3))
  y=Twin(Config(seed=1,nmonths=3))
  y.fx[2]=10000
  x.run();y.run()
  # At most previous published month's rate enters decisions.
  self.assertAlmostEqual(x.real_output,y.real_output)
  self.assertAlmostEqual(x.trade_value,y.trade_value)
if __name__=='__main__':unittest.main()