import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from economy_v8 import Config,Economy,A,MODES
class StockFlowTest(unittest.TestCase):
    def test_industry_coefficients(self):
        self.assertTrue((A>=0).all());self.assertLess(A.sum(axis=0).max(),.4)
    def test_credit_creates_deposit(self):
        z=Economy(Config());before=z.fd.copy();loan=z.fl.copy()
        z.loan_f(np.array([0]),np.array([8.]))
        self.assertAlmostEqual(z.fd[0]-before[0],8)
        self.assertAlmostEqual(z.fl[0]-loan[0],8);z.audit()
    def test_program_off(self):
        x=Economy(Config('synergy_cyber','baseline',9)).run(36)
        self.assertEqual(x.iloc[-1].escrow_unlocked,0)
    def test_program_equal_matching(self):
        x=Economy(Config('synergy_cyber','escrow_demand',9)).run(120).iloc[-1]
        self.assertGreater(x.escrow_unlocked,0)
        self.assertAlmostEqual(x.escrow_unlocked,x.bank_matched,places=8)
    def test_double_entry_and_resources(self):
        for mode in MODES:
            e=Economy(Config(mode,'escrow_demand',77,True));p=e.run(120)
            self.assertLess(p.financial_residual.max(),1e-5)
            self.assertLess(p.input_output_residual.max(),1e-8)
            self.assertLessEqual(p.iloc[-1].real_final_sales,p.iloc[-1].final_supply+1e-6)
    def test_no_future_macro_in_controller(self):
        import inspect
        s=inspect.getsource(Economy)
        self.assertNotIn('real_gdp_yoy_percent',s)
        self.assertNotIn('cpi_dec_to_dec_percent',s)
if __name__=='__main__': unittest.main()