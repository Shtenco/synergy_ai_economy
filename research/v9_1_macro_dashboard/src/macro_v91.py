"""V9.1 multi-dimensional macro observations and synthetic traded equity + ROW trade.

Research-only extensions of V9. Numeric values/foreign exchange denominated in a
SHARED EXPERIMENTAL NUMERAIRE; not RUB/GDP/MOEX/official foreign reserves.
The stock market is actual shareholder-to-shareholder settlement with bounded
order crossing; mark prices are synthetic and never treated as created output.
ROW reserve assets are acquired only by surrendering actual exported goods, and
can finance physical imports. EAEU inter-central-bank claims are not reserves.
"""
from __future__ import annotations
import numpy as np
from twin import Twin, Config, N, S, A, TECH_COST, CONSUMPTION_MIX, COUNTRIES

class MacroTwin(Twin):
    def __init__(self,cfg:Config):
        super().__init__(cfg)
        self.base_price=self.price.copy()
        self.row_reserves=np.zeros(N)
        self.world_imports=np.zeros((N,S))
        self.world_exports=np.zeros((N,S))
        self.world_import_payments=np.zeros(N)
        self.world_export_receipts=np.zeros(N)
        self.this_gva_real=np.zeros(N)
        self.this_gva_nominal=np.zeros(N)
        self.gva_real_history=[]
        self.gva_nominal_history=[]
        self.cpi_history=[]
        self.indexed_real_deposit=[]
        self.monthly_consump=[]
        self.equity_shares=np.full((N,self.hh_count,S),100./self.hh_count)
        self.shares_issued=np.full((N,S),100.)
        self.equity_price=np.maximum(.05,(self.fdep+self.capacity*TECH_COST[None,:]-self.loan)/100.)
        self.equity_initial_cap=float((self.equity_price*self.shares_issued).sum())
        self.equity_cum_volume=0.
        self.equity_trades=0
        self.dividends_paid=0.
        # Fixed individual investor beliefs; identical across economic institutions.
        self.investor_bias=np.tile(np.linspace(-.22,.22,self.hh_count),(N,1))
        self.row_goods_sold=0.
        self.row_goods_bought=0.

    def _audit(self):
        if not hasattr(self,'world_imports'):
            return super()._audit()
        if not np.all(np.isfinite(self.stock)) or np.min(self.stock)<-1e-7: raise AssertionError('physical inventory negative')
        if min(np.min(self.fdep),np.min(self.hdep),np.min(self.hbalance),np.min(self.breserve))<-1e-7:raise AssertionError('cash negative')
        if not np.allclose(self.hbalance.sum(axis=1),self.hdep,atol=3e-7):raise AssertionError('household bank deposits unequal')
        if not np.allclose(self.bank_equity(),self.expected_bank_equity,atol=3e-7):raise AssertionError('bank equity unequal')
        if not np.allclose(self.cb_gov_assets+self.cb_external+self.bank_refi,self.breserve+self.cb_gov,atol=3e-7):raise AssertionError('CB double entry unequal')
        if not np.allclose(self.govdebt,self.bank_bonds+self.cb_gov_assets,atol=3e-7):raise AssertionError('public debt unequal')
        lhs=self.initial_physical+self.total_produced+self.world_imports.sum(axis=0)
        rhs=self.stock.sum(axis=0)+self.total_used
        if not np.allclose(lhs,rhs,atol=3e-7):raise AssertionError(('physical conservation with trade',lhs-rhs))
        if abs(self.unlocked-self.matched)>1e-7 or self.unlocked>self.issued+1e-7:raise AssertionError('escrow')
        if abs(self.capital_cost-self.capital_commissioned-self.capital_abandoned-sum(p.funded for p in self.projects))>1e-7:raise AssertionError('WIP')
        if np.min(self.row_reserves)<-1e-7:raise AssertionError('reserves below zero')
        if not np.allclose(self.equity_shares.sum(axis=1),self.shares_issued,atol=3e-7):raise AssertionError('share conservation')

    def _world_trade(self):
        """Physical ROW transactions; no exogenous money giveaways or free FX reserves.
        Exports require available stock, import payments require reserves + deposits.
        Outside-world account is the opposite side of every currency claim.
        """
        for c in range(N):
            for s in (0,2,3):
                # Exogenous small ROW customer demand is equal-law across modes.
                demand=.11*self.capacity[c,s]
                surplus=max(0.,self.stock[c,s]-.52*self.capacity[c,s]-10.)
                units=min(demand,surplus)
                if units<=1e-9:continue
                value=units*self.price[c,s]
                self.stock[c,s]-=units;self.total_used[s]+=units
                self.world_exports[c,s]+=units
                self.row_reserves[c]+=value
                self.cb_external[c]+=value
                self.breserve[c]+=value;self.fdep[c,s]+=value
                self.world_export_receipts[c]+=value
                self.row_goods_sold+=units
            # Import goods can only be bought with previously acquired external assets.
            for s in (0,2,3):
                target=.16*self.capacity[c,s]+7.
                gap=max(0.,target-self.stock[c,s])
                unit_price=self.price[c,s]*1.06
                affordable=max(0.,self.fdep[c,s]-28.)/unit_price
                qty=min(gap,affordable,self.row_reserves[c]/unit_price,2.5)
                if qty<=1e-9:continue
                value=qty*unit_price
                self.stock[c,s]+=qty
                self.world_imports[c,s]+=qty
                self.row_reserves[c]-=value
                self.cb_external[c]-=value
                self.breserve[c]-=value;self.fdep[c,s]-=value
                self.world_import_payments[c]+=value
                self.row_goods_bought+=qty

    def _trade(self):
        super()._trade()
        self._world_trade()

    def _produce(self):
        q=super()._produce()
        self.last_production=q.copy()
        # Sector gross value minus physically used intermediate value (same month).
        input_nom=np.einsum('cs,is,ci->c',q,A,self.price)
        gross_nom=np.sum(q*self.price,axis=1)
        self.this_gva_nominal=gross_nom-input_nom
        inp_real=np.einsum('cs,is,ci->c',q,A,self.base_price)
        self.this_gva_real=np.sum(q*self.base_price,axis=1)-inp_real
        if min(self.this_gva_real.min(),self.this_gva_nominal.min())<-1e-6:raise AssertionError('negative sectoral value addition')
        return q

    def _exchange(self):
        """Cash-settled secondary equity transactions + financed cash distributions.
        A price mark is NOT issued money, aggregate GDP, nor realizable HardNAV.
        """
        if self.c.mode=='soviet': return # public enterprises, no freely traded shareholder equity
        for c in range(N):
            for s in range(S):
                previous=self.equity_price[c,s]
                # Fully observable enterprise balance data; price = private quote,
                # not a forecast, not actual exchange capitalization.
                intrinsic=max(.12,(self.fdep[c,s]+self.capacity[c,s]*TECH_COST[s]-self.loan[c,s])/100.)
                valuation=intrinsic*(1+self.investor_bias[c])
                buyers=list(np.argsort(-valuation))
                sellers=list(np.argsort(valuation))
                buyer_index=0;seller_index=0;last_price=previous
                steps=0
                while buyer_index<len(buyers) and seller_index<len(sellers) and steps<12:
                    b=int(buyers[buyer_index]);seller=int(sellers[seller_index])
                    if b==seller or valuation[b]<valuation[seller]*1.015:
                        buyer_index+=1;continue
                    if valuation[b]<previous*.98 or valuation[seller]>previous*1.02:
                        break
                    price=float(np.clip((valuation[b]+valuation[seller])/2,previous*.985,previous*1.015))
                    qty=min(self.equity_shares[c,seller,s]*.08,max(0.,self.hbalance[c,b]-2.)/price,.85)
                    if qty<.0001:
                        buyer_index+=1;seller_index+=1;continue
                    # Equal-and-opposite bank-account movement, ownership conserved.
                    self.hbalance[c,b]-=price*qty
                    self.hbalance[c,seller]+=price*qty
                    self.equity_shares[c,b,s]+=qty
                    self.equity_shares[c,seller,s]-=qty
                    self.equity_cum_volume+=price*qty
                    self.equity_trades+=1
                    last_price=price
                    buyer_index+=1;seller_index+=1;steps+=1
                self.equity_price[c,s]=last_price
                if self.t%3==0:
                    # Distribution funded by actual firm deposits (NOT a fictional gain).
                    dividend=min(max(0.,self.fdep[c,s]-185.)*.002,1.5)
                    if dividend>0:
                        self.fdep[c,s]-=dividend
                        additions=self.equity_shares[c,:,s]*(dividend/100.)
                        self.hbalance[c]+=additions;self.hdep[c]+=additions.sum()
                        self.dividends_paid+=dividend
                        self.equity_price[c,s]=max(.02,self.equity_price[c,s]-dividend/100.)
        self._audit()

    def step(self):
        before_cons=self.real_cons
        before_import_payments=float(self.world_import_payments.sum())
        row=super().step()
        self._exchange()
        self.gva_nominal_history.append(float(self.this_gva_nominal.sum()))
        self.gva_real_history.append(float(self.this_gva_real.sum()))
        basket=(self.price*CONSUMPTION_MIX).sum(axis=1)/(self.base_price*CONSUMPTION_MIX).sum(axis=1)
        basket_total=float(basket.mean())
        self.cpi_history.append(basket_total)
        wealth_cash=(self.hbalance/np.maximum(basket[:,None],1e-9))
        wealth_equity=(np.einsum('chs,cs->ch',self.equity_shares,self.equity_price)/basket[:,None] if self.c.mode!='soviet' else np.zeros_like(wealth_cash))
        self.indexed_real_deposit.append(float(np.median(wealth_cash)))
        cons_month=self.real_cons-before_cons
        self.monthly_consump.append(cons_month)
        def yoy(ar):
            return float(ar[-1]/ar[-13]-1) if len(ar)>=13 and ar[-13]!=0 else float('nan')
        def flow_yoy(ar):
            return float(sum(ar[-12:])/sum(ar[-24:-12])-1) if len(ar)>=24 and sum(ar[-24:-12])>0 else float('nan')
        annual_real_gva=sum(self.gva_real_history[-12:]);annual_nom_gva=sum(self.gva_nominal_history[-12:]);
        capratio=self.bank_equity()/np.maximum(1,self.breserve+self.bank_bonds+self.loan.sum(axis=1)+self.bank_equity_claim)
        # Relative fragility across histories, not empirical default probability.
        monthly_cpi_log=np.diff(np.log(self.cpi_history[-13:])) if len(self.cpi_history)>1 else np.array([])
        equity_cap=float((self.equity_price*self.shares_issued).sum())
        row.update({
           'gva_month_nominal_proxy':self.gva_nominal_history[-1],
           'gva_month_real_proxy':self.gva_real_history[-1],
           'gva_annual_real_proxy':annual_real_gva,
           'gva_annual_nominal_proxy':annual_nom_gva,
           'real_gva_yoy_pct':100*flow_yoy(self.gva_real_history),
           'nominal_gva_yoy_pct':100*flow_yoy(self.gva_nominal_history),
           'cpi_fixed_basket':basket_total,
           'cpi_yoy_pct':100*yoy(self.cpi_history),
           'cpi_monthly_pct':100*(self.cpi_history[-1]/self.cpi_history[-2]-1) if len(self.cpi_history)>1 else float('nan'),
           'inflation_instability_ann_pct':float(np.std(monthly_cpi_log)*np.sqrt(12)*100) if len(monthly_cpi_log)>=8 else float('nan'),
           'median_real_household_deposits':float(np.median(wealth_cash)),
           'mean_real_household_deposits':float(np.mean(wealth_cash)),
           'median_real_household_wealth_with_equity':float(np.median(wealth_cash+wealth_equity)),
           'household_total_wealth_real_mark':float((wealth_cash+wealth_equity).sum()),
           'household_wealth_gini_with_equity':self._gini(wealth_cash+wealth_equity),
           'low_liquid_buffer_pct':float(np.mean(wealth_cash<2.)),
           'monthly_real_consumption_per_hh':cons_month/(N*self.hh_count),
           'annual_real_consumption_per_hh':sum(self.monthly_consump[-12:])/(N*self.hh_count),
           'equity_price_index_100':(equity_cap/self.equity_initial_cap*100. if self.c.mode!='soviet' else float('nan')),
           'equity_market_cap_mark':(equity_cap if self.c.mode!='soviet' else float('nan')),
           'equity_cumulative_cash_turnover':self.equity_cum_volume,
           'equity_cumulative_trades':self.equity_trades,
           'equity_cumulative_cash_dividends':self.dividends_paid,
           'row_external_reserves_asset':float(self.row_reserves.sum()),
           'row_export_receipts':float(self.world_export_receipts.sum()),
           'row_import_payments':float(self.world_import_payments.sum()),
           'row_reserves_import_months':float(self.row_reserves.sum()/max(1e-6,self.world_import_payments.sum()/(self.t+1))) if self.world_import_payments.sum()>0 else float('nan'),
           'domestic_gross_supply_share_proxy_pct':100*float((self.this_gva_nominal+np.einsum('cs,is,ci->c',self.last_production,A,self.price)).sum())/max(1e-9,float((self.this_gva_nominal+np.einsum('cs,is,ci->c',self.last_production,A,self.price)).sum())+float(self.world_import_payments.sum())-before_import_payments),
           'row_import_units_cum':float(self.world_imports.sum()),
           **{'reserve_claim_'+COUNTRIES[c]:float(self.row_reserves[c]) for c in range(N)},
           **{'gva_real_'+COUNTRIES[c]:float(self.this_gva_real[c]) for c in range(N)},
           **{'cpi_'+COUNTRIES[c]:float(basket[c]) for c in range(N)},
           'row_export_units_cum':float(self.world_exports.sum()),
           'domestic_output_by_sector_agri':float(self.total_produced[0]),
           'domestic_output_by_sector_energy':float(self.total_produced[1]),
           'domestic_output_by_sector_materials':float(self.total_produced[2]),
           'domestic_output_by_sector_machinery':float(self.total_produced[3]),
           'domestic_output_by_sector_construction':float(self.total_produced[4]),
           'domestic_output_by_sector_logistics':float(self.total_produced[5]),
           'bank_min_capital_ratio':float(np.min(capratio)),
           'bank_nonpositive_capital_count':int((self.bank_equity()<=0).sum()),
           'government_debt_to_annual_gva_proxy_pct':100*float(self.govdebt.sum())/max(1e-9,annual_nom_gva),
           'financial_net_external_assets_ex_EAEU':float(self.row_reserves.sum()),
           'physical_audit_error':float(np.max(abs(self.initial_physical+self.total_produced+self.world_imports.sum(axis=0)-self.stock.sum(axis=0)-self.total_used))),
           'real_gva_positive':bool(self.gva_real_history[-1]>0),
           'stock_market_status':('NOT_APPLICABLE_PUBLIC_OWNERSHIP' if self.c.mode=='soviet' else 'SYNTHETIC_SETTLED'),
           'foreign_reserve_status':'SYNTHETIC_ROW_TRADE'
        })
        # Bank deposits reflect the actually paid dividends and market settlement.
        row['household_deposits']=float(self.hdep.sum())
        row['firm_deposits']=float(self.fdep.sum())
        row['m2_proxy']=float(self.hdep.sum()+self.fdep.sum())
        row['household_wealth_gini']=float(self._gini(self.hbalance))
        self.history[-1]=row
        self._audit()
        return row