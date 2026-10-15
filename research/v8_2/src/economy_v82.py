"""V8.2 banking extension: genuine cash (M0), household loan creation and repayment.
Research prototype. IO coefficients remain *illustrative* until Rosstat originals are ingested.
No OOS observations are read by the simulator, ever.
"""
from dataclasses import dataclass
import numpy as np
from economy_v81_legacy import Economy as Legacy,Config,MODES,POLICIES,gini

@dataclass
class Config82(Config):
    household_credit_rate:float=.0
    household_amortization:float=.016
    target_cash_ratio:float=.15
    min_capital_ratio:float=.085

class Economy82(Legacy):
    def __init__(self,cfg:Config82):
        self.cash_household=np.zeros(400)
        self.cbr_currency=0.
        self.credit_h_issued=0.
        self.credit_h_repaid=0.
        self.credit_h_interest=0.
        self.bank_credit_veto=0.
        super().__init__(cfg)
        # Convert existing deposits to cash without creating wealth or bank equity.
        self.convert_cash(cfg.target_cash_ratio * np.maximum(self.hh,0))
        self.audit()
    def financial_net(self):
        base=super().financial_net()
        return float(base+self.cash_household.sum()-self.cbr_currency)
    def convert_cash(self,amount):
        v=np.clip(np.asarray(amount,float),0,np.maximum(self.hh,0))
        # Preserve a bank settlement reserve floor. Reserve -> currency issued by CBR.
        bank_total=np.bincount(self.hbank,weights=v,minlength=4)
        scal=np.minimum(1,np.maximum(self.reserve-80,0)/np.maximum(bank_total,1e-12))
        v*=scal[self.hbank]
        self.hh-=v;self.cash_household+=v;self.cbr_currency+=float(v.sum())
        self.reserve-=np.bincount(self.hbank,weights=v,minlength=4)
        self.audit()
    def deposit_cash(self,amount):
        v=np.clip(np.asarray(amount,float),0,self.cash_household)
        self.cash_household-=v;self.hh+=v;self.cbr_currency-=float(v.sum())
        self.reserve+=np.bincount(self.hbank,weights=v,minlength=4)
        self.audit()
    def money_supply_proxy(self):
        return float(self.d_by_bank().sum()+self.cbr_currency)
    def household_credit_cycle(self,m):
        # Lend to households using PREVIOUS period bank capitalization & credit limits.
        # Lends create a deposit AND an equal loan claim. No external bailout/fresh bank equity.
        debt=np.maximum(self.hl,0)
        payment=np.minimum(debt*self.c.household_amortization,np.maximum(self.hh,0)*.025)
        self.hl-=payment;self.hh-=payment
        self.credit_h_repaid+=float(payment.sum())
        # Interest debits customers, retained by banks (therefore raises bank equity).
        interest=np.minimum(np.maximum(self.hh,0)*.015,debt*np.minimum(.22,self.rate+.055)/12)
        self.hh-=interest;self.credit_h_interest+=float(interest.sum())
        if self.c.household_credit_rate<=0 or m>=120:
            self.audit();return
        gross_income_est=np.maximum(.1,self.skill*1.45)
        ceiling=np.maximum(0,27*gross_income_est-debt)
        requested=np.minimum(self.hh*self.c.household_credit_rate,ceiling)
        requested=np.minimum(requested,2.7*gross_income_est)
        eq=self.equity();assets=np.maximum(self.reserve+self.bank_bonds+self.l_by_bank(),1)
        # Each bank's *aggregate* lending veto, not per-account credit grants.
        headroom=np.maximum(0,eq/self.c.min_capital_ratio-assets)
        # Liquidity buffer and reserve maintenance reduce, rather than create, credit.
        available=np.maximum(self.reserve-85,0)*1.25
        limit=np.minimum(headroom,available)
        desired=np.bincount(self.hbank,weights=requested,minlength=4)
        multiplier=np.minimum(1,limit/np.maximum(desired,1e-12))
        issued=requested*multiplier[self.hbank]
        self.bank_credit_veto+=float(requested.sum()-issued.sum())
        self.hl+=issued;self.hh+=issued;self.credit_h_issued+=float(issued.sum())
        self.audit()
    def month_step(self,m):
        self.month=m
        self.household_credit_cycle(m)
        super().month_step(m)
        # Slow portfolio preference for currency cash, no artificial cash creation.
        total=self.cash_household.sum()+self.hh.sum()+self.fd.sum()
        actual=self.cbr_currency/max(total,1)
        if actual<self.c.target_cash_ratio-.012:
            self.convert_cash(.018*np.maximum(self.hh,0))
        elif actual>self.c.target_cash_ratio+.012:
            self.deposit_cash(.018*self.cash_household)
        q=self.history[-1]
        q.update(m0_currency=float(self.cbr_currency),deposit_money=float(self.d_by_bank().sum()),
                 m2_proxy=float(self.money_supply_proxy()), household_loans=float(self.hl.sum()),
                 household_credit_issued=float(self.credit_h_issued), household_credit_repaid=float(self.credit_h_repaid),
                 household_interest_paid=float(self.credit_h_interest),credit_veto=float(self.bank_credit_veto),
                 bank_min_capratio=float(np.min(self.equity()/np.maximum(self.reserve+self.bank_bonds+self.l_by_bank(),1))))
        self.audit()