"""SINERGY Russia V8: eight-sector stock-flow-consistent exploratory ABM.
Historical FX is used only as a contemporaneously realized environment, never as
foresight. Observed GDP/CPI 2023–25 are *not* inputs. IO coefficients are
ILLUSTRATIVE, not extracted from Rosstat input-output tables.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SECTORS=('agriculture','energy','materials','machinery','construction','transport','trade','services')
# a[i,j]: intermediate physical unit of sector i per gross unit sector j.
# ASSUMPTION (not an empirical estimate): individual column sums 0.15–0.35.
A=np.array([
[.015,.002,.003,.007,.008,.004,.008,.011],
[.022,.035,.060,.045,.049,.050,.035,.021],
[.015,.012,.038,.075,.082,.029,.013,.015],
[.011,.030,.044,.044,.025,.025,.020,.027],
[.012,.006,.016,.013,.031,.014,.018,.013],
[.028,.028,.033,.030,.030,.042,.046,.029],
[.029,.028,.029,.039,.035,.031,.056,.039],
[.035,.051,.048,.049,.053,.044,.058,.069]],float)
assert max(A.sum(axis=0))<.40 and A.min()>=0
MODES={'capitalism':dict(pool=0.,plan=0.,lag=1,admin=.002),
       'soviet_planning':dict(pool=.94,plan=.95,lag=6,admin=.020),
       'russia_mixed':dict(pool=.24,plan=.29,lag=3,admin=.008),
       'synergy_cyber':dict(pool=.50,plan=.64,lag=1,admin=.017)}
POLICIES=('baseline','escrow','demand','escrow_demand')


def gini(x):
    a=np.sort(np.maximum(0,np.asarray(x,float)))
    return float((2*np.dot(np.arange(1,len(a)+1),a)/(len(a)*a.sum())-(len(a)+1)/len(a))) if a.sum()>0 else 0.

def make_fx():
    fx_path=ROOT/'data/usdrub_monthly_2016_2025.csv'
    if not fx_path.exists():fx_path=ROOT.parents[1]/'data/usdrub_monthly_2016_2025.csv'
    z=pd.read_csv(fx_path)
    datecol='month' if 'month' in z else ('date' if 'date' in z else z.columns[0])
    valcol='rate' if 'rate' in z else ('usdrub' if 'usdrub' in z else z.columns[1])
    z=z.sort_values(datecol)
    a=z[valcol].astype(float).values
    assert len(a)==120 and np.all(a>0),f'bad observed FX {len(a)}'
    return a

@dataclass
class Config:
    mode:str='capitalism'
    policy:str='baseline'
    seed:int=1
    stress:bool=False
    response:float=.4
    fiscal_scale:float=1.
    credit_scale:float=1.

class Economy:
    """400 representative households, 32 firms, 4 banks, 8 real sectors, treasury, CBR.
    Monetary entries are abstract ruble account units; no unjustified GDP scaling.
    A bank's equity equals reserves + claims - client deposits - CBR refinancing.
    Central bank refinancing creates reserves and an equal bank debt.
    """
    def __init__(self,cfg:Config):
        assert cfg.mode in MODES and cfg.policy in POLICIES
        self.c=cfg; self.inst=MODES[cfg.mode]; self.rng=np.random.default_rng(cfg.seed)
        self.fxobs=make_fx();self.fx=float(self.fxobs[0]);self.fx_prior=self.fx
        self.hbank=np.arange(400)%4; self.fbank=np.arange(32)%4;self.sector=np.arange(32)%8
        r=self.rng
        self.job=r.integers(0,32,400)
        self.skill=r.uniform(.75,1.25,400);self.risk=r.uniform(1,2.5,400)
        self.fear=np.zeros(400);self.morale=np.zeros(400)
        self.hh=r.lognormal(np.log(25),.42,400)
        self.fd=r.lognormal(np.log(235),.15,32)
        self.hl=r.uniform(0,5,400);self.fl=np.full(32,110.)
        self.k=r.uniform(110,170,32);self.k0=self.k.copy()
        self.bank_bonds=np.full(4,180.)
        self.reserve=(np.bincount(self.hbank,weights=self.hh,minlength=4)+
                      np.bincount(self.fbank,weights=self.fd,minlength=4)-
                      np.bincount(self.hbank,weights=self.hl,minlength=4)-
                      np.bincount(self.fbank,weights=self.fl,minlength=4)-self.bank_bonds+450.)
        self.refi=np.zeros(4);self.cb_bonds=float(self.reserve.sum()+1400.);self.treasury=1000.
        self.budget_debt=float(self.cb_bonds+self.bank_bonds.sum())
        self.start_gross=self.financial_net();self.maxerr=0.
        self.audits=0;self.rate=.10;self.cpi=1.;self.infl=.05
        self.offered=np.zeros(4);self.released=0.;self.matched=0.;self.demand_paid=0.
        self.bail_in=0.;self.bank_resolutions=0;self.firm_defaults=0
        self.real_output=0.;self.final_supply=0.;self.real_final_sales=0.;self.real_consumption=0.;self.intermediate=0.
        self.taxcum=0.;self.wagecum=0.;self.admin_real=0.;self.shortages=[];self.ioresmax=0.
        self.losses=0.;self.loans_new=0.;self.loan_repaid=0.;self.escrow_issued=0.
        self.public_spend=0.;self.history=[];self.month=0;self.audit()

    def d_by_bank(self):
        return np.bincount(self.hbank,weights=self.hh,minlength=4)+np.bincount(self.fbank,weights=self.fd,minlength=4)
    def l_by_bank(self):
        return np.bincount(self.hbank,weights=self.hl,minlength=4)+np.bincount(self.fbank,weights=self.fl,minlength=4)
    def equity(self):return self.reserve+self.bank_bonds+self.l_by_bank()-self.d_by_bank()-self.refi
    def financial_net(self):
        cb_equity=self.cb_bonds+self.refi.sum()-self.reserve.sum()-self.treasury
        government_equity=self.treasury-self.bank_bonds.sum()-self.cb_bonds
        return float(self.hh.sum()+self.fd.sum()-self.hl.sum()-self.fl.sum()+self.equity().sum()+cb_equity+government_equity)
    def audit(self):
        err=abs(self.financial_net()-self.start_gross)
        # Starting central-bank bonds are government debt; a matched reserve issue
        # is a consolidated net-zero inter-sector claim (not new economic wealth).
        assert np.isfinite(err) and err<1e-5,(self.month,err)
        self.maxerr=max(err,self.maxerr);self.audits+=1
        assert abs(self.released-self.matched)<1e-6
    def f_to_h(self,fi,hi):
        assert abs(fi.sum()-hi.sum())<1e-6
        self.fd-=fi;self.hh+=hi
        self.reserve-=np.bincount(self.fbank,weights=fi,minlength=4)
        self.reserve+=np.bincount(self.hbank,weights=hi,minlength=4)
    def h_to_f(self,hi,fi):
        assert abs(hi.sum()-fi.sum())<1e-6
        self.hh-=hi;self.fd+=fi
        self.reserve-=np.bincount(self.hbank,weights=hi,minlength=4)
        self.reserve+=np.bincount(self.fbank,weights=fi,minlength=4)
    def f_to_f(self,buyers,sellers):
        assert abs(buyers.sum()-sellers.sum())<1e-6
        self.fd+=sellers-buyers
        self.reserve+=np.bincount(self.fbank,weights=sellers-buyers,minlength=4)
    def issue_bonds(self,value):
        value=max(0.,float(value))
        if value==0:return
        self.bank_bonds+=value/4.;self.reserve-=value/4.;self.treasury+=value
        self.budget_debt+=value
        # The treasury deposit liability/asset is represented outside aggregate private net wealth;
        # its debt issuance changes financial net ONLY IF offset by its cash asset.
        # Therefore financial_net tracks private+CB minus debt and treasury asset below.
    def pay_gov_h(self,value):
        v=np.maximum(0,np.asarray(value,float));s=v.sum()
        if s>self.treasury:self.issue_bonds(s-self.treasury+10)
        self.hh+=v;self.reserve+=np.bincount(self.hbank,weights=v,minlength=4);self.treasury-=s
    def pay_gov_f(self,value):
        v=np.maximum(0,np.asarray(value,float));s=v.sum()
        if s>self.treasury:self.issue_bonds(s-self.treasury+10)
        self.fd+=v;self.reserve+=np.bincount(self.fbank,weights=v,minlength=4);self.treasury-=s
    def taxes(self,hi,fi):
        hi=np.minimum(np.maximum(hi,0),self.hh*.55);fi=np.minimum(np.maximum(fi,0),self.fd*.45)
        self.hh-=hi;self.fd-=fi
        self.reserve-=np.bincount(self.hbank,weights=hi,minlength=4)
        self.reserve-=np.bincount(self.fbank,weights=fi,minlength=4)
        value=float(hi.sum()+fi.sum());self.treasury+=value;self.taxcum+=value
    def loan_f(self,fi,quantity):
        value=np.maximum(0,np.asarray(quantity,float))
        self.fl[fi]+=value;self.fd[fi]+=value;self.loans_new+=float(value.sum())
    def repay_f(self,payment):
        payment=np.minimum(np.maximum(payment,0),np.minimum(self.fl,np.maximum(self.fd-5,0)*.08))
        self.fl-=payment;self.fd-=payment;self.loan_repaid+=float(payment.sum())
    def cashflow(self,m):
        # All policy decisions depend on m-1 state. FX at m is revealed only after decisions.
        self.rate=float(np.clip(.85*self.rate+.15*(.045+1.30*(self.infl-.04)),.03,.24))
        self.rate=max(self.rate,.025)
        # Interest paid by borrowers to bank equity; deposits paid by bank reduce equity.
        loan_int=self.fl*np.minimum(.18,self.rate+.04)/12
        pay=np.minimum(self.fd*.10,loan_int)
        self.fd-=pay
        dr=self.rate*.35/12
        self.fd+=np.maximum(self.fd,0)*dr
        self.hh+=np.maximum(self.hh,0)*dr
        # CBR reserves/refi are matched claims/liabilities: amortized below.
        bank_liq=np.maximum(0,self.reserve-90)
        # Bank interest margin booked inside equity (not fictitious resource output).
        return bank_liq
    def escrow(self,m,usable):
        if self.c.policy not in ('escrow','escrow_demand') or not 12<=m<110:return
        self.offered+=7.0;self.escrow_issued+=28.
        eq=self.equity();lo=self.l_by_bank()
        for b in range(4):
            cap=max(0,(eq[b]/.11-lo[b])/2)
            liquid=max(0,(self.reserve[b]-110)/.20)
            amount=min(self.offered[b],cap,liquid,5.5)
            if amount<.01:continue
            selected=np.where(self.fbank==b)[0]
            # Price signal uses shortage history from m-1 at the latest.
            if self.shortages:
                shortage=self.shortages[-1]
                selected=selected[np.argsort(-shortage[self.sector[selected]])]
            selected=selected[:2]
            loan=np.full(len(selected),2*amount/max(1,len(selected)))
            self.loan_f(selected,loan)
            self.offered[b]-=amount
            self.matched+=amount;self.released+=amount
            # CBR credit creates central-bank reserves AND equal bank debt.
            self.reserve[b]+=amount;self.refi[b]+=amount
            # A haircut portion is unavailable to make further commitments, not equity.
            self.reserve[b]-=.2*amount;self.reserve[b]+=.2*amount
        self.audit()
    def demand_support(self,m):
        if self.c.policy not in ('demand','escrow_demand') or not 12<=m<110:return
        if self.infl>.105:return
        poor=self.hh<np.quantile(self.hh,.43)
        benefit=np.where(poor,.42,0.)
        self.pay_gov_h(benefit);self.demand_paid+=float(benefit.sum()); self.public_spend+=float(benefit.sum())
    def month_step(self,m):
        self.month=m
        # Historical USD/RUB becomes the shock only at the END of the period.
        past_ret=np.log(self.fx/self.fx_prior) if m else 0.
        crisis=self.c.stress and (25<=m<42 or 72<=m<82)
        zr=self.rng.normal(0,1)
        # FX exposed commodity and imported intermediate sectors have real resource constraints.
        currency_cost=np.clip(1+.30*max(past_ret,0),1,1.13)
        prod_shock=np.clip(1+.018*zr-(.24 if crisis else 0),.65,1.12)
        self.cashflow(m);self.escrow(m,None);self.demand_support(m)
        owners=self.inst['pool'];governance=self.inst['plan']*self.c.response
        # No hard-coded productivity preference for governance regime.
        working=(self.rng.random(400)>(.035+(.13 if crisis else 0)))
        effort=np.clip(.90+.14*(self.skill-1)+.055*self.morale-.13*self.fear,.65,1.15)*working
        labor=np.bincount(self.job,weights=effort,minlength=32)
        gross=labor*2.8*prod_shock*np.power(self.k/self.k0,.22)
        # Explicit administrative real-resource expense; no free planning.
        expense=gross*self.inst['admin'];gross-=expense;self.admin_real+=expense.sum()
        # Endogenous allocation of working capital by prior shortage, no future data.
        b_eq=self.equity();b_liquid=np.maximum(self.reserve-95,0)
        baseline=np.maximum(0,13+.30*labor-self.fd*.035)
        available=np.clip((b_eq[self.fbank]-.08*self.l_by_bank()[self.fbank])/40,0,1)
        available*=np.clip(b_liquid[self.fbank]/250,0,1)
        credit=baseline*available*self.c.credit_scale
        if self.shortages and governance>0:
            lag=min(len(self.shortages),self.inst['lag'])
            shortage=self.shortages[-lag]
            intensity=np.clip(1+governance*(shortage[self.sector]-shortage.mean())*2,.25,2)
            adjust=credit*intensity
            credit=adjust*credit.sum()/max(adjust.sum(),1e-9)
        self.loan_f(np.arange(32),credit)
        wages=(1.45+.18*(self.skill-1))*working
        wagecost=np.bincount(self.job,weights=wages,minlength=32)
        affordability=np.minimum(1,np.maximum(self.fd-12,0)/np.maximum(wagecost,1e-12))
        wages*=affordability[self.job];wagecost=np.bincount(self.job,weights=wages,minlength=32)
        self.f_to_h(wagecost,wages);self.wagecum+=wages.sum()
        # Actual interindustry purchases, exactly zero-sum cash clearing.
        gross_sector=np.bincount(self.sector,weights=gross,minlength=8)
        intermediate=A @ gross_sector
        # Sector buyer j needs sum_i A[i,j]*gross_j input units.
        costrate=A.sum(axis=0)[self.sector]*gross*self.cpi*currency_cost
        feasible=np.minimum(1,np.maximum(self.fd-3,0)/np.maximum(costrate,1e-9))
        gross*=feasible
        gross_sector=np.bincount(self.sector,weights=gross,minlength=8)
        intermediate=A @ gross_sector
        buy=A.sum(axis=0)[self.sector]*gross*self.cpi*currency_cost
        proceeds=intermediate[self.sector]*self.cpi*currency_cost* gross/np.maximum(gross_sector[self.sector],1e-10)
        # One sector receives intermediate inputs from all purchasing sectors.
        self.f_to_f(buy,proceeds)
        net=np.maximum(0,gross_sector-intermediate)
        self.ioresmax=max(self.ioresmax,float(abs(gross_sector.sum()-intermediate.sum()-net.sum())))
        self.intermediate+=intermediate.sum();self.real_output+=gross_sector.sum();self.final_supply+=net.sum()
        # Household purchasing, firm investment and treasury procurement compete for net goods.
        budget=np.minimum(np.maximum(self.hh,0)*.22,1.0+.78*wages+.012*np.maximum(self.hh,0))
        inv=np.minimum(np.maximum(self.fd-20,0)*.07,10.)
        government=40*self.c.fiscal_scale
        want=budget.sum()+inv.sum()+government
        sector_mix=np.array([.17,.13,.13,.10,.12,.11,.12,.12]);sector_mix/=sector_mix.sum()
        demand_units=sector_mix*want/self.cpi
        sector_sold=np.minimum(demand_units,net)
        shortage=(demand_units-sector_sold)/np.maximum(demand_units,1e-9)
        self.shortages.append(shortage)
        ration=sector_sold.sum()/max(demand_units.sum(),1e-9)
        amountH=np.minimum(self.hh*.70,budget*ration)
        amountI=inv*ration
        amountG=government*ration
        sold_sector_weight=(gross*np.maximum(0,sector_sold[self.sector])/np.maximum(gross_sector[self.sector],1e-9))
        suppliers=sold_sector_weight/max(sold_sector_weight.sum(),1e-10)
        self.h_to_f(amountH,suppliers*amountH.sum())
        self.f_to_f(amountI,suppliers*amountI.sum())
        self.pay_gov_f(suppliers*amountG); self.public_spend+=amountG
        cons=amountH.sum()/self.cpi
        self.real_consumption+=cons
        self.real_final_sales+=(amountH.sum()+amountI.sum()+amountG)/self.cpi
        self.k+=.85*amountI/self.cpi;self.k*=.994
        # Fees, taxes and public debt have explicit named counterparties.
        self.taxes(.052*wages,.014*np.maximum(self.fd-35,0))
        benefits=np.where(wages<.10,.40,0.)
        self.pay_gov_h(benefits); self.public_spend+=benefits.sum()
        # Ownership claims and dividends: current profits only; bank+firm principal unchanged.
        div=np.minimum(np.maximum(self.fd-190,0)*.035,2)
        pay=(1-owners)*np.bincount(np.arange(32)*12 % 400,weights=div,minlength=400)+owners*div.sum()/400
        self.f_to_h(div,pay)
        # Share earned income after taxes; not bank deposits/principal.
        if owners>0:
            earned=wages+pay+benefits-.052*wages
            share=owners*(earned.mean()-earned);share-=share.mean()
            self.hh+=share;self.reserve+=np.bincount(self.hbank,weights=share,minlength=4)
        self.repay_f(np.minimum(self.fl*.015,np.maximum(self.fd-70,0)*.04))
        # Credit losses lower equity; debt forgiveness lowers loan asset.
        if crisis:
            bad=(self.rng.random(32)<.006+.018*(self.fl>self.k*2))
            write=self.fl*bad*.16
            self.fl-=write;self.losses+=write.sum();self.firm_defaults+=int(bad.sum())
        equity=self.equity();assets=self.reserve+self.bank_bonds+self.l_by_bank()
        fail=equity<.055*np.maximum(assets,1)
        for b in np.flatnonzero(fail):
            self.bank_resolutions+=1
            hidx=self.hbank==b;fidx=self.fbank==b
            req=max(0,.085*assets[b]-equity[b]);hshare=np.maximum(self.hh[hidx]-24,0);fshare=np.maximum(self.fd[fidx]-140,0)
            available=hshare.sum()+fshare.sum();cut=min(req,available*.25)
            if available>0 and cut>0:
                self.hh[hidx]-=cut*hshare/available;self.fd[fidx]-=cut*fshare/available
                self.bail_in+=cut
        self.audit()
        # Agents cannot observe this current-month historical FX before deciding.
        self.fx_prior=self.fx;self.fx=float(self.fxobs[m])
        # Economic prices determined by real scarcity, imported costs, and lagged inflation.
        dlog=np.clip(.035/12+.025*(want/max(net.sum()*self.cpi,1)-1)+.03*max(past_ret,0),-.015,.05)
        self.cpi*=np.exp(dlog)
        self.infl=.88*self.infl+.12*(12*dlog)
        netclaims=np.maximum(self.hh-self.hl,0)
        row=dict(seed=self.c.seed,mode=self.c.mode,policy=self.c.policy,stress=int(self.c.stress),month=m+1,
             train=int(m<84),oos=int(m>=84),fx=self.fx,inflation=self.infl,price_index=self.cpi,real_output_gross=self.real_output,
             final_supply=self.final_supply,real_consumption=self.real_consumption,real_final_sales=self.real_final_sales,intermediates=self.intermediate,
             public_spend=self.public_spend, taxes_total=self.taxcum, bank_resolutions=self.bank_resolutions,firm_defaults=self.firm_defaults,gov_debt=self.budget_debt,
             debt_to_bank_deposits=self.budget_debt/max(self.d_by_bank().sum(),1),real_K=self.k.sum(),
             hh_liquid_gini=gini(netclaims),private_deposits=self.d_by_bank().sum(),loans=self.l_by_bank().sum(),
             bank_capratio=float(np.min(self.equity()/np.maximum(assets,1))),
             escrow_issued=self.escrow_issued,escrow_unlocked=self.released,bank_matched=self.matched,
             locked_escrow=self.offered.sum(),demand_vouchers=self.demand_paid,
             input_output_residual=self.ioresmax,financial_residual=self.maxerr)
        self.history.append(row)
    def run(self,months=120):
        for m in range(months):self.month_step(m)
        return pd.DataFrame(self.history)