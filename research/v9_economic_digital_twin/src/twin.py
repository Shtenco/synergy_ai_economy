"""SINERGY V9: physically stock-conserving, financially reconciled EAEU toy ABM.
NOT a calibrated digital twin or empirical input-output matrix.
Same technologies and proposal budget for all institutional policies.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np

COUNTRIES=('RU','KZ','BY','AM','KG')
GOODS=('agri','electricity','materials','machinery','construction','logistics','commerce','services')
MODES=('capitalism','soviet','russia_mixed','synergy')
POLICIES=('baseline','escrow','demand','escrow_demand')
N=5;S=8
# RESEARCH ASSUMPTION: rows physical commodity inputs, columns output industry.
A=np.array([
 [0,.000,.014,.015,.017,.003,.012,.009],
 [.032,0,.052,.050,.038,.042,.025,.020],
 [.012,.026,.000,.072,.095,.021,.019,.012],
 [.016,.040,.065,.000,.036,.027,.014,.020],
 [.009,.018,.023,.028,0,.020,.013,.017],
 [.018,.022,.037,.043,.039,0,.042,.028],
 [.025,.030,.031,.032,.020,.030,0,.030],
 [.032,.041,.047,.043,.047,.024,.046,0]],float)
assert np.max(A.sum(axis=0))<.36 and np.all(A>=0)
# Fixed technology shared by all institutions. Changes in capacity only after construction.
TECH_COST=np.array([5.0,5.0,6.0,6.0,4.5,5.5,4.5,4.0])
CAPEX_MIX=np.array([.07,.10,.26,.23,.20,.09,.0,.05])
SKILL_DEMAND=np.array([[.7,.25,.05],[.35,.4,.25],[.35,.4,.25],[.15,.4,.45],
                       [.5,.4,.1],[.5,.4,.1],[.5,.4,.1],[.25,.35,.4]])
CONSUMPTION_MIX=np.array([.29,.09,.05,.04,.06,.08,.13,.26])
ARCHETYPES=(
 ('PowerGen',1,4),('Grid',1,4),('MineralRefining',2,5),('AdvancedMaterials',2,6),
 ('PowerElectronics',3,5),('DataCenterInfra',7,6),('Compute',7,6),('AICloud',7,5),
 ('IndustrialAI',3,4),('Robotics',3,7),('MachineTools',3,5),('SemiDesign',7,8),
 ('Logistics',5,4),('STEM',7,6),('AgroGenomics',0,6),('SeedProduction',0,5),
 ('PrecisionAg',0,5),('AgroRobotics',0,7),('FoodProcessing',0,4),
 ('WaterIrrigation',0,6),('RuralAIClusters',7,7),('UrbanRenovation',4,5),
 ('RuralHousing',4,5),('ConstructionTech',4,5))

@dataclass
class Project:
    country:int
    sector:int
    archetype:int
    outstanding:np.ndarray
    total_cost:float
    funded:float
    launched:int
    months_worked:int=0
    delay:int=0
    bank_equity_at_risk:float=0.

@dataclass
class Config:
    mode:str='synergy'
    policy:str='escrow_demand'
    seed:int=1
    nmonths:int=120
    stress:bool=False
    planning_weight:float=.5
    inspection_budget:int=8
    funding_cap:float=8.0
    subsidy:float=12.0

class Twin:
 def __init__(self,cfg:Config):
    assert cfg.mode in MODES and cfg.policy in POLICIES
    self.c=cfg;self.rng=np.random.default_rng(cfg.seed)
    self.fx=self._load_fx()
    self.t=0;self.stock=np.tile(np.array([24.,18.,24.,18.,21.,18.,19.,22.]),(N,1))
    self.capacity=np.tile(np.array([16.,18.,15.,12.,14.,11.,13.,17.]),(N,1))
    # Cross-country endowments are illustrative and shared across ALL policies.
    self.capacity*=np.array([[1.0,1.3,1.25,1.35,1.0,1.12,1.0,1.0],
                             [1.2,1.15,1.15,.68,.8,1.15,1.0,.9],
                             [1.0,.95,1.10,1.25,1.15,.9,1.0,.92],
                             [.85,.9,.70,.75,1.05,.7,.9,1.1],
                             [1.15,.95,.80,.62,1.03,.7,1.03,.85]])
    self.grid=np.array([15.,14.,12.,10.,11.])
    self.water=np.array([14.,13.,13.,10.,11.])
    self.route=np.full(N,24.)
    self.labor=np.tile(np.array([70.,68.,46.]),(N,1))
    self.available_labor=self.labor.copy()
    self.price=np.tile(np.array([1.1,1.15,1.3,1.6,1.5,1.2,1.,1.25]),(N,1))
    self.hdep=np.full(N,600.)
    self.hh_count=60
    hh=self.rng.lognormal(0,.45,size=(N,self.hh_count))
    self.hbalance=hh/hh.sum(axis=1,keepdims=True)*600.
    self.hh_sector=np.tile(np.arange(self.hh_count)%S,(N,1))
    self.hh_skill=self.rng.integers(0,3,size=(N,self.hh_count))
    pref=self.rng.lognormal(0,.20,size=(N,self.hh_count,S))*CONSUMPTION_MIX[None,None,:]
    self.hh_pref=pref/pref.sum(axis=2,keepdims=True)
    self.fdep=np.full((N,S),220.)
    self.loan=np.full((N,S),75.)
    self.bank_equity_claim=np.zeros(N)
    self.bank_bonds=np.full(N,150.)
    self.bank_refi=np.zeros(N)
    self.breserve=np.full(N,2010.)
    self.cb_gov=np.zeros(N)
    self.cb_gov_assets=self.breserve.copy()
    self.cb_external=np.zeros(N)
    self.govdebt=self.bank_bonds+self.cb_gov_assets
    self.escrow_locked=np.zeros(N)
    self.issued=0.;self.unlocked=0.;self.matched=0.;self.fiscal=0.
    self.expected_bank_equity=self.bank_equity().copy()
    self.initial_physical=self.stock.sum(axis=0).copy()
    self.spoiled=np.zeros(S)
    self.total_produced=np.zeros(S);self.total_used=np.zeros(S)
    self.consumed=np.zeros((N,S));self.real_output=0.;self.real_cons=0.
    self.trade_units=0.;self.trade_value=0.;self.cumulative_labor=0.
    self.project_started=0;self.project_completed=0;self.project_abandoned=0
    self.capital_cost=0.;self.capital_commissioned=0.;self.capital_abandoned=0.
    self.projects=[];self.completed_history=[];self.events=[]
    self.short=np.zeros((N,S));self.prev_short=np.zeros((N,S))
    self.history=[];self.failures=0;self.bank_veto=0;self.escrow_veto=0
    self._audit()

 @staticmethod
 def _load_fx():
    file=Path(__file__).resolve().parents[1]/'data'/'usdrub_2016_2025.csv'
    if not file.is_file():return np.full(120,np.nan)
    with file.open(newline='',encoding='utf-8') as fh:
      items=list(csv.DictReader(fh))
    a=np.array([float(x['usdrub_official_last_publication_month']) for x in items])
    if len(a)!=120 or np.min(a)<=0:raise ValueError('USD/RUB needs 120 observed monthly values')
    return a

 @staticmethod
 def _gini(x):
    a=np.sort(np.maximum(0.,np.asarray(x).ravel()))
    if not a.sum():return 0.
    n=len(a)
    return float((2*np.dot(np.arange(1,n+1),a)/(n*a.sum()))-(n+1)/n)

 def bank_equity(self):
    return self.breserve+self.bank_bonds+self.loan.sum(axis=1)+self.bank_equity_claim-self.hdep-self.fdep.sum(axis=1)-self.bank_refi
 def _audit(self):
    if not np.all(np.isfinite(self.stock)) or np.min(self.stock)<-1e-7:raise AssertionError('Negative physical stock')
    if np.min(self.fdep)<-1e-7 or np.min(self.hdep)<-1e-7 or np.min(self.hbalance)<-1e-7:raise AssertionError('Negative deposits')
    if not np.allclose(self.hbalance.sum(axis=1),self.hdep,atol=3e-7):raise AssertionError('Household individual vs bank deposits')
    if np.min(self.breserve)<-1e-7:raise AssertionError('Negative bank reserve')
    if not np.allclose(self.bank_equity(),self.expected_bank_equity,atol=3e-7):
      raise AssertionError(('bank equity reconciliation',self.bank_equity()-self.expected_bank_equity))
    if not np.allclose(self.cb_gov_assets+self.cb_external+self.bank_refi,self.breserve+self.cb_gov,atol=3e-7):
      raise AssertionError('CB assets != liabilities')
    if not np.allclose(self.govdebt,self.bank_bonds+self.cb_gov_assets,atol=3e-7):
      raise AssertionError('Government bond holders != debt')
    if not np.allclose(self.initial_physical+self.total_produced,self.stock.sum(axis=0)+self.total_used,atol=3e-7):
      raise AssertionError('Goods produced/used/held imbalance')
    if self.unlocked>self.issued+1e-7 or abs(self.unlocked-self.matched)>1e-7:
      raise AssertionError('Escrow invalid matched condition')
    if min(self.capital_commissioned,self.capital_abandoned)<-1e-7:raise AssertionError('Negative capital')
    if abs(self.capital_cost-self.capital_commissioned-self.capital_abandoned-sum(p.funded for p in self.projects))>1e-7:
      raise AssertionError('Capital project WIP reconciliation')

 def _transfer(self,a,b,amount):
    """Physical transfer with matched seller/buyer deposits; common numeraire accounting.
    Seller and buyer are (country, sector), no cross-border central-bank creation."""
    ca,sa=a;cb,sb=b
    x=max(0.,min(float(amount),self.fdep[cb,sb],self.stock[ca,sa]*self.price[ca,sa]))
    if x<1e-10:return 0.
    units=x/self.price[ca,sa]
    if ca!=cb:
      x=min(x,max(0,self.breserve[cb]-25.));units=x/self.price[ca,sa]
    self.fdep[ca,sa]+=x;self.fdep[cb,sb]-=x
    self.breserve[ca]+=x;self.breserve[cb]-=x
    if ca!=cb:
      self.cb_external[ca]+=x;self.cb_external[cb]-=x
    self.stock[ca,sa]-=units;self.stock[cb,sa]+=units
    if ca!=cb:self.trade_value+=x;self.trade_units+=units
    return units

 def _bond_issue(self,c,x):
    x=max(0.,float(x))
    x=min(x,max(0,self.breserve[c]-20))
    self.breserve[c]-=x;self.bank_bonds[c]+=x
    self.cb_gov[c]+=x;self.govdebt[c]+=x
    return x
 def _voucher(self):
    if self.c.policy not in ('demand','escrow_demand') or self.t<12:return
    # Identical conditional fiscal instrument for all four institutional systems.
    for c in range(N):
      if self.short[c].mean()<.018 and self.hdep[c]>270:continue
      amount=self._bond_issue(c,self.c.subsidy)
      self.cb_gov[c]-=amount;self.hdep[c]+=amount;self.breserve[c]+=amount
      poor=self.hbalance[c]<np.quantile(self.hbalance[c],.4)
      shares=poor.astype(float);shares/=shares.sum()
      self.hbalance[c]+=amount*shares
      self.fiscal+=amount

 def _finance(self,c,s,amount):
    """CB escrow requires contemporaneous, real bank equity subscription 1:1.
    Bank purchase of newly issued firm equity credits a deposit against a booked
    risky equity claim; it consumes RWA headroom, not pre-existing outside cash.
    CB loan creates reserves+refi; second project tranche is a real corporate loan.
    No balances generated merely by offering/locking a contingent escrow ceiling."""
    if self.c.policy not in ('escrow','escrow_demand') or self.t<12:return 0.
    x=max(0.,min(amount,self.escrow_locked[c]))
    assets=self.breserve[c]+self.bank_bonds[c]+self.loan[c].sum()+self.bank_equity_claim[c]
    head=max(0.,self.bank_equity()[c]/.12-assets)
    x=min(x,head/2,max(0,self.breserve[c]-80.)/.15,7.0)
    if x<1e-9:self.escrow_veto+=1;return 0.
    # Independently authorized bank share subscription => traceable risk exposure.
    self.fdep[c,s]+=x;self.bank_equity_claim[c]+=x
    # Targeted second tranche: credit asset and matching deposit.
    self.fdep[c,s]+=x;self.loan[c,s]+=x
    # Secured CB refinancing: equal reserve and debt, no CB grant.
    self.breserve[c]+=x;self.bank_refi[c]+=x
    self.unlocked+=x;self.matched+=x;self.escrow_locked[c]-=x
    return 2*x

 def _new_project(self,c,s,archetype,total_cost,bank_equity_at_risk=0.):
    total_cost=float(total_cost)
    if self.fdep[c,s]<total_cost+25:return False
    if total_cost<2:return False
    budget=CAPEX_MIX*total_cost
    obj=Project(c,s,archetype,budget,total_cost,0.,self.t,bank_equity_at_risk=bank_equity_at_risk)
    self.projects.append(obj);self.project_started+=1
    return True

 def _scores(self):
    # Every regime sees same lagged physical shortages, market prices,
    # input-output technical coefficients, deposits; no direct productivity bonus.
    lag=self.prev_short.copy()
    profits=(self.fdep-100.)/np.maximum(self.capacity*35,1.)
    crit=np.linalg.solve(np.eye(S)-A,np.ones(S));crit/=crit.mean()
    leverage=(crit-1)[None,:]
    scarcity=lag/np.maximum(.025,lag.mean())
    w=self.c.planning_weight
    if self.c.mode=='capitalism':
      return (1-w)*profits+w*(.9*scarcity-.10)
    if self.c.mode=='soviet':
      return (1-w)*scarcity+w*leverage-.14
    if self.c.mode=='russia_mixed':
      return (1-w)*profits+w*(.55*scarcity+.30*leverage)-.035
    return (1-w)*profits+w*(scarcity+leverage)-.11

 def _offer_escrow(self):
    if self.c.policy in ('escrow','escrow_demand') and self.t>=12:
       self.escrow_locked+=self.c.funding_cap
       self.issued+=self.c.funding_cap*N

 def _project_select(self):
    scores=self._scores()
    # Same exactly 8 feasible proposals considered per period regardless of mode.
    # Different priority rules only; no privileged data or free real assets.
    proposals=[(float(scores[c,s]),c,s) for c in range(N) for s in range(S)]
    proposals.sort(key=lambda z:(-z[0],z[1],z[2]))
    for _,c,s in proposals[:self.c.inspection_budget]:
      if sum(p.country==c and p.sector==s for p in self.projects)>1:continue
      typ=(int(self.t)+int(s)*3+int(c))%len(ARCHETYPES)
      # Archetypes are mapped to the model's actual sector for consistent capacity.
      candidates=[i for i,(_,ss,_) in enumerate(ARCHETYPES) if ss==s]
      typ=next((i for i in candidates if i>=typ),candidates[0]) if candidates else 12
      cost=TECH_COST[s]*(1.5+.5*(self.t%3))
      financed=self._finance(c,s,min(self.c.funding_cap,cost/2))
      if self.fdep[c,s]<cost+25:continue
      self._new_project(c,s,typ,cost,bank_equity_at_risk=financed/2)

 def _procure(self):
    # Each unfinished investment requires real physical units, cash and skill-hours.
    keep=[]
    for p in self.projects:
      c,s=p.country,p.sector
      stage_bill=np.minimum(p.outstanding,CAPEX_MIX*p.total_cost/3.)
      labour_need=1.8
      if self.available_labor[c,2]<labour_need:
        p.delay+=1
      elif np.all(self.stock[c]>=stage_bill-1e-8) and self.fdep[c,s]>=float((stage_bill*self.price[c]).sum())+5:
        # Payments go to actual firms supplying the physical construction goods.
        for i,u in enumerate(stage_bill):
          cost=u*self.price[c,i]
          self.fdep[c,s]-=cost;self.fdep[c,i]+=cost
          self.stock[c,i]-=u;self.total_used[i]+=u
        self.available_labor[c,2]-=labour_need
        self.capital_cost+=float(stage_bill.sum())
        p.outstanding-=stage_bill
        p.funded+=float(stage_bill.sum());p.months_worked+=1
      else:p.delay+=1
      lead=ARCHETYPES[p.archetype][2]
      if np.max(p.outstanding)<1e-9 and self.t-p.launched>=lead:
        # Deployment increases specific physical capacity only; not current goods.
        name=ARCHETYPES[p.archetype][0]
        if name=='Grid':self.grid[c]+=1.1*p.total_cost
        elif name=='WaterIrrigation':self.water[c]+=.60*p.total_cost
        elif name=='STEM':self.labor[c,2]+=.6*p.total_cost
        elif name=='Logistics':self.route[c]+=.55*p.total_cost
        else:self.capacity[c,s]+=p.total_cost/TECH_COST[s]
        self.project_completed+=1;self.capital_commissioned+=p.funded
        self.completed_history.append((self.t,c,s,p.total_cost))
      elif self.t-p.launched>lead+11:
        # Unsatisfied project write-off: sunk physical goods are NOT returned.
        self.capital_abandoned+=p.funded;self.project_abandoned+=1
        # Explicit bank first-loss from equity stakes (amount limited by its book value).
        haircut=min(self.bank_equity_claim[c],p.bank_equity_at_risk,p.funded*.9)
        self.bank_equity_claim[c]-=haircut;self.expected_bank_equity[c]-=haircut
      else:keep.append(p)
    self.projects=keep

 def _trade(self):
    # Exporting region sells actual inventory to another, subject to route capacity.
    for s in (0,1,2,3,4,5,7):
      for b in range(N):
        threshold=max(12.,self.capacity[b,s]*.55)
        deficit=max(0.,threshold-self.stock[b,s])
        if deficit<=.02:continue
        exporters=sorted(range(N),key=lambda a:self.stock[a,s],reverse=True)
        for a in exporters:
          if a==b:continue
          surplus=max(0.,self.stock[a,s]-max(10.,self.capacity[a,s]*.5))
          units=min(deficit,surplus,self.route[a],self.route[b],self.stock[a,5]*7)
          if units<=.02:continue
          cost=units*self.price[a,s]
          # FX shock comes from the latest **already published** month, never t.
          fx_old=self.fx[max(0,self.t-1)] if self.t else np.nan
          fx_older=self.fx[max(0,self.t-2)] if self.t>1 else np.nan
          fx_fee=min(.045,.15*abs(np.log(fx_old/fx_older))) if np.isfinite(fx_old) and np.isfinite(fx_older) else 0.
          fee=cost*fx_fee
          if self.fdep[b,s]<cost+fee+3:continue
          units=self._transfer((a,s),(b,s),cost)
          # Border payment friction credited to an actual logistics service.
          fee=min(fee,self.fdep[b,s]);self.fdep[b,s]-=fee;self.fdep[b,5]+=fee
          # Real freight fuel/service consumed at exporter.
          transport=min(self.stock[a,5],units*.04)
          self.stock[a,5]-=transport;self.total_used[5]+=transport
          self.route[a]-=units;self.route[b]-=units
          deficit-=units
          if deficit<=.02:break

 def _produce(self):
    """Simultaneous multi-firm physical clearing within one monthly accounting period.
    For each commodity i: opening_inventory_i + newly_produced_i >=
    intermediate_inputs_i. Intra-month supplier deliveries permit viable Leontief
    loops without inventing stocks; all goods still counted once.
    """
    capacity=self.capacity.copy()
    skilled=np.maximum(0.,self.available_labor.copy())
    z=self.rng.normal(0,.045,size=(N,S))
    gross=np.zeros((N,S))
    for c in range(N):
      need=capacity[c]@SKILL_DEMAND
      labor_scale=min([1.]+[skilled[c,j]/max(need[j],1e-9) for j in range(3)])
      desired=capacity[c]*max(0.,min(1.,labor_scale))*np.clip(1+z[c],.65,1.25)
      desired[1]=min(desired[1],self.grid[c])
      desired[0]=min(desired[0],self.water[c]/.9)
      if self.c.stress and 24<=self.t%120<=45:desired*=.80
      # Working-capital gate: no operation if producer cannot finance input purchases.
      wc=np.maximum(.01,A.T@self.price[c])
      desired=np.minimum(desired,np.maximum(0.,self.fdep[c]-10)/np.maximum(wc,1e-9))
      q=desired.copy()
      # Solve physical feasibility under nonnegative inventory and input recipes.
      # Downstream users scale in response to shortages in their input markets.
      for attempt in range(64):
        req=A@q
        available=self.stock[c]+q
        deficit=req-available
        if np.max(deficit)<1e-8:break
        for i in np.where(deficit>1e-8)[0]:
          consumers=np.where(A[i]>0)[0]
          if len(consumers)==0:continue
          fac=max(0.,min(1.,available[i]/max(req[i],1e-10)))
          q[consumers]*=fac
      consumed=A@q
      if np.min(self.stock[c]+q-consumed)<-1e-7:raise AssertionError('Leontief physical infeasibility')
      self.stock[c]+=q-consumed
      self.total_produced+=q
      self.total_used+=consumed
      gross[c]=q
      self.available_labor[c]-=q@SKILL_DEMAND
      self.cumulative_labor+=float(q.sum())
      # Actual purchases of intermediate goods transfer deposits among firms.
      for j in range(S):
        for i in range(S):
          value=A[i,j]*q[j]*self.price[c,i]
          value=min(value,max(0.,self.fdep[c,j]-2.))
          self.fdep[c,j]-=value;self.fdep[c,i]+=value
    self.real_output+=gross.sum()
    return gross

 def _households(self):
    # 300 independently budgeted households (60 per country); firms pay workers.
    for c in range(N):
      wage=np.minimum(.20*self.fdep[c],self.capacity[c]*.21)
      for sector in range(S):
        subset=np.where(self.hh_sector[c]==sector)[0]
        weights=1.+.25*self.hh_skill[c,subset]
        weights=weights/weights.sum()
        self.hbalance[c,subset]+=wage[sector]*weights
      self.fdep[c]-=wage;self.hdep[c]+=wage.sum()
      individual_budget=np.minimum(self.hbalance[c]*.085,1.0+.015*self.hbalance[c])
      money_demand=individual_budget[:,None]*self.hh_pref[c]
      available=self.stock[c]*self.price[c]
      desired=money_demand.sum(axis=0)
      ration=np.minimum(1.,available/np.maximum(desired,1e-10))
      purchase=money_demand*ration[None,:]
      cost_by_household=purchase.sum(axis=1)
      cost_by_sector=purchase.sum(axis=0)
      self.hbalance[c]-=cost_by_household
      self.hdep[c]-=cost_by_household.sum()
      self.fdep[c]+=cost_by_sector
      sold=cost_by_sector/self.price[c]
      self.stock[c]-=sold;self.total_used+=sold
      self.consumed[c]+=sold;self.real_cons+=sold.sum()
      if self.c.mode in ('soviet','russia_mixed','synergy'):
        # All regimes redistribute a transparent share of already earned firm cash.
        share={'soviet':.032,'russia_mixed':.014,'synergy':.020}[self.c.mode]
        pool=share*sum(self.fdep[c])
        self.fdep[c]-=share*self.fdep[c];self.hdep[c]+=pool
        if self.c.mode=='soviet':
          shares=np.full(self.hh_count,1/self.hh_count)
        else:
          shares=1/np.maximum(1,self.hbalance[c])
          shares=shares/shares.sum()
        self.hbalance[c]+=pool*shares

 def _loans_interest(self):
    # Loan interest transfers deposits to bank equity; no fictitious money.
    for c in range(N):
      charged=np.minimum(self.fdep[c]*.08,self.loan[c]*.004)
      self.fdep[c]-=charged
      self.expected_bank_equity[c]+=charged.sum()
      repayment=np.minimum(self.loan[c]*.003,np.maximum(0,self.fdep[c]-25.)*.004)
      self.fdep[c]-=repayment;self.loan[c]-=repayment

 def step(self):
    self._audit()
    self._offer_escrow()
    self._voucher()
    self._loans_interest()
    self._project_select()
    self.available_labor=self.labor.copy()
    # Infrastructure physical limits reset for month: transport is flow capacity,
    # not a one-off nonrenewable inventory.
    route_installed=self.route.copy()
    self._trade()
    self.route=route_installed
    self._produce()
    self._households()
    self._procure()
    # Perishable inventories and power loss; expiring goods are not bank assets.
    perish=self.stock[:,0]*.025
    power_loss=self.stock[:,1]*.20
    self.stock[:,0]-=perish;self.stock[:,1]-=power_loss
    self.total_used[0]+=perish.sum();self.total_used[1]+=power_loss.sum()
    self.spoiled[0]+=perish.sum();self.spoiled[1]+=power_loss.sum()
    # Identical causal adjustment of posted prices to observed stock scarcity.
    self.price*=np.exp(np.clip(.025*(self.short-.055),-.012,.012))
    self.prev_short=self.short.copy()
    self.short=np.maximum(0,(self.capacity*.40-self.stock))/np.maximum(1,self.capacity*.40)
    self._audit()
    row=dict(month=self.t+1,mode=self.c.mode,policy=self.c.policy,seed=self.c.seed,
      stress=int(self.c.stress),physical_output=self.real_output,consumption=self.real_cons,
      household_deposits=float(self.hdep.sum()),firm_deposits=float(self.fdep.sum()),
      m2_proxy=float(self.hdep.sum()+self.fdep.sum()),
      bank_equity=float(self.bank_equity().sum()),gov_debt=float(self.govdebt.sum()),
      fiscal=float(self.fiscal),escrow_issued=float(self.issued),escrow_released=float(self.unlocked),
      matched_bank_equity=float(self.matched),construction_started=self.project_started,
      commissioned=self.project_completed,abandoned=self.project_abandoned,
      capex_commissioned=float(self.capital_commissioned),capex_written_off=float(self.capital_abandoned),
      wip_paid=float(sum(p.funded for p in self.projects)),trade_units=float(self.trade_units),
      trade_value=float(self.trade_value),producer_shortage=float(self.short.mean()),
      energy_generation=float(self.total_produced[1]),
      usdrub_observed=float(self.fx[self.t]) if self.t<len(self.fx) else float("nan"),
      physical_audit_error=float(np.max(abs(self.initial_physical+self.total_produced-self.stock.sum(axis=0)-self.total_used))),
      bank_audit_error=float(np.max(abs(self.bank_equity()-self.expected_bank_equity))),
      foreign_cb_net_assets=float(self.cb_external.sum()),
      household_wealth_gini=float(self._gini(self.hbalance.ravel())),
      households=int(self.hh_count*N))
    self.history.append(row);self.t+=1
    return row
 def run(self,months=None):
    n=self.c.nmonths if months is None else months
    for _ in range(n):self.step()
    return self.history