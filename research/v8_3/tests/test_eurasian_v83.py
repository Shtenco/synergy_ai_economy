import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from economy_v83 import Economy83,Config83
from eurasian_factory import ARCHETYPES,ProjectFactory
from economy_v81_legacy import A,MODES

class TestEurasianPipeline(unittest.TestCase):
 def test_24_real_eiiom_sector_names(self):
  self.assertEqual(len(ARCHETYPES),24)
  self.assertTrue(all(0<=x[1]<=7 for x in ARCHETYPES))
 def test_all_modes_share_same_physical_factory(self):
  for mode in MODES:
   e=Economy83(Config83(mode=mode,seed=17,enable_projects=True))
   self.assertEqual(e.factory.completion_horizon,6)
   np.testing.assert_allclose(e.factory.A,A)
 def test_escrow_and_goods_ledger(self):
  e=Economy83(Config83(mode='synergy_cyber',policy='escrow_demand',seed=89,stress=True,
                       credit_scale=2.,fiscal_scale=7.5))
  e.run(42);e.factory.audit()
  self.assertLess(e.maxerr,1e-7)
  self.assertLess(e.ioresmax,1e-8)
  self.assertAlmostEqual(e.released,e.matched,places=9)
 def test_no_free_commissioned_capital(self):
  e=Economy83(Config83(mode='capitalism',seed=18))
  e.run(24)
  q=e.factory
  self.assertAlmostEqual(q.capital_spent,q.commissioned_capital+q.written_off+sum(p.paid_capital for p in q.projects),places=6)
 def test_project_latency(self):
  e=Economy83(Config83(mode='synergy_cyber',seed=10))
  e.month_step(0)
  self.assertEqual(e.factory.completed,0)
  self.assertGreater(e.factory.started,0)
 def test_off_mode_matches_legacy(self):
  from economy_v82 import Economy82,Config82
  opts=dict(mode='russia_mixed',seed=12,policy='baseline',stress=False)
  a=Economy83(Config83(**opts,enable_projects=False));b=Economy82(Config82(**opts))
  da=a.run(15);db=b.run(15)
  np.testing.assert_allclose(da.real_final_sales,db.real_final_sales,atol=1e-7)
 def test_no_future_fx_in_decisions(self):
  # Existing model reveals fxobs[m] only after month_step financial/project choices.
  import inspect
  from economy_v81_legacy import Economy
  src=inspect.getsource(Economy.month_step)
  self.assertLess(src.index('self.escrow(m,None)'),src.index('self.fx=float(self.fxobs[m])'))

if __name__=='__main__':unittest.main()