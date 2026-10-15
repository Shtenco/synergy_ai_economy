"""EIIOM-inspired Real Economy Project Factory, bounded prototype.
Canonical source: Shtenco/synergy_eurasian (EIIOM-v2 runnable LP;
EIIOM-v4 MILP specification). This is newly implemented reference logic,
NOT the full EIIOM-v4, not empirical Rosstat IO coefficients.

Invariants:
  - WIP receives ONLY physical capital actually invested and removed from K.
  - Project completion releases at most previously deposited WIP into K.
  - Scarce capacity, trained labour and explicit lag govern commissioning.
  - Neither escrow issuance nor modelled future GVA creates free real capital.
  - All economic modes access the SAME project recipe, costs and shock law;
    only project-selection rules vary.
"""
from dataclasses import dataclass
import numpy as np

SECTORS=('agriculture','energy','materials','machinery','construction','transport','trade','services')
# Actual EIIOM-v2 sector names, eight-sector mapping is RESEARCH ASSUMPTION.
ARCHETYPES=[
 ('PowerGen',1,3,1.12),('Grid',1,4,1.08),('MineralRefining',2,4,1.04),
 ('AdvancedMaterials',2,5,1.18),('PowerElectronics',3,5,1.14),
 ('DataCenterInfra',7,5,1.28),('Compute',7,5,1.35),('AICloud',7,4,1.18),
 ('IndustrialAI',3,4,1.22),('Robotics',3,6,1.32),('MachineTools',3,5,1.12),
 ('SemiDesign',7,7,1.30),('Logistics',5,3,1.05),('STEM',7,6,1.20),
 ('AgroGenomics',0,6,1.23),('SeedProduction',0,5,1.05),
 ('PrecisionAg',0,4,1.10),('AgroRobotics',0,6,1.22),('FoodProcessing',0,3,1.05),
 ('WaterIrrigation',0,5,1.08),('RuralAIClusters',7,7,1.25),
 ('UrbanRenovation',4,4,1.05),('RuralHousing',4,4,1.07),
 ('ConstructionTech',4,5,1.12)]

@dataclass
class WIP:
    project_id:int
    firm:int
    paid_capital:float
    maturity:int
    starting_month:int
    delayed:int=0

class ProjectFactory:
    def __init__(self, mode, seed, A, w_shortage=.7, completion_horizon=6):
        self.mode=mode;self.seed=seed;self.w_shortage=w_shortage
        self.projects=[];self.cancelled=0.;self.completed=0.;self.started=0.
        self.delay_total=0;self.gate_denials=0;self.month_history=[]
        self.capital_spent=0.;self.commissioned_capital=0.;self.written_off=0.
        self.training_capital=0.;self.training_completions=0
        self.completion_horizon=completion_horizon
        self.A=np.asarray(A,float)
        # Actual IO matrix is still unknown; multipliers are ILLUSTRATIVE.
        self.leontief=np.linalg.solve(np.eye(8)-self.A,np.ones(8))
        self.sector_by_firm=np.arange(32)%8
        self.role_by_firm=np.array([x%len(ARCHETYPES) for x in range(32)])

    def _observables(self,e,month):
        ksec=np.bincount(e.sector,weights=e.k,minlength=8)
        work=np.bincount(e.job,weights=e.skill,minlength=32)
        skill=np.bincount(e.sector,weights=work,minlength=8)
        shortages=np.mean(np.array(e.shortages[-min(6,len(e.shortages)):]),axis=0) if e.shortages else np.zeros(8)
        shortage_old=e.shortages[-min(6,len(e.shortages))] if e.shortages else np.zeros(8)
        profit=np.maximum(0,e.fd)/(np.maximum(5,e.fl)+np.maximum(5,e.k))
        # All these are observable at time t or before. No future inputs.
        return ksec,skill,shortages,shortage_old,profit

    def rank_firms(self,e,month):
        ksec,skill,short,old,profit=self._observables(e,month)
        si=e.sector
        # Same observable economic fundamentals for all systems.
        cash=np.log1p(np.maximum(e.fd,0))
        quality=profit/max(np.std(profit),1e-5)
        bottleneck=(short-short.mean())/max(np.std(short),.08)
        critical=self.leontief/np.mean(self.leontief)-1
        risk=np.maximum(0,.13-e.equity()[e.fbank]/np.maximum(e.reserve[e.fbank]+e.bank_bonds[e.fbank]+e.l_by_bank()[e.fbank],1))
        # Sector resource constraints enter selection as an observed commissioning risk.
        need=np.array([ARCHETYPES[int(self.role_by_firm[i])][3] for i in range(32)])
        res=np.minimum.reduce([ksec[1]/max(np.mean(ksec),1e-6)*np.ones(32),
                               ksec[2]/max(np.mean(ksec),1e-6)*np.ones(32),
                               ksec[5]/max(np.mean(ksec),1e-6)*np.ones(32)])
        gate_risk=np.maximum(0,need-res)
        if self.mode=='capitalism':
            score=1.3*quality+.25*(cash-np.mean(cash))-.8*gate_risk
        elif self.mode=='soviet_planning':
            score=1.9*old[si]+.15*critical[si]-.55*gate_risk
        elif self.mode=='russia_mixed':
            score=.78*quality+.45*old[si]+.15*critical[si]-.65*gate_risk
        elif self.mode=='synergy_cyber':
            # Explicit bounded cybernetic algorithm, NO productivity multiplier.
            score=(1.0-self.w_shortage)*quality + self.w_shortage*(1.3*bottleneck[si]+.50*critical[si]) - .8*gate_risk - 1.7*risk
        else:raise ValueError(self.mode)
        # Deterministic tie breaks are shared across modes.
        return np.argsort(-score+1e-7*np.arange(32))

    def collect_real_investment(self,e,kbefore,m):
        # The mother model already purchased FINAL REAL goods and recognized new K.
        # Remove the same amount from productive K and place into nonoperational WIP.
        realized=np.maximum(0,e.k-kbefore*.994)
        invest_real=float(realized.sum())
        if invest_real<1e-12:return 0.
        to_stage=realized*.70
        e.k-=to_stage
        budget=float(to_stage.sum())
        ranked=self.rank_firms(e,m)
        # Physical investment buys capital goods before a project can be commissioned.
        # Changing firm destination is physical capital transfer, not free new capital.
        allocated=np.zeros(32)
        available=budget
        for fi in ranked[:12]:
            room=max(0,3.5+.015*e.k[fi]-sum(p.paid_capital for p in self.projects if p.firm==fi))
            amt=min(available,room)
            if amt>.000001:
                allocated[fi]=amt;available-=amt
            if available<=1e-9:break
        # Remaining WIP capital allocated without preference if 12 slots saturate.
        if available>0:
            allocated[ranked[0]]+=available
        assert abs(allocated.sum()-budget)<1e-7
        for fi,amount in enumerate(allocated):
            if amount>1e-9:
                archetype_id=int(self.role_by_firm[fi]); base_lag=ARCHETYPES[archetype_id][2]
                self.projects.append(WIP(archetype_id,fi,float(amount),m+base_lag,m))
                self.started+=1
        self.capital_spent+=budget
        return budget

    def _can_complete(self,e,p,m):
        ksec=np.bincount(e.sector,weights=e.k,minlength=8)
        mean=float(np.mean(ksec));difficulty=ARCHETYPES[p.project_id][3]
        # Capital capacity is ex-ante available, NOT made up by printing.
        ratios={'energy':ksec[1]/max(mean,1e-6),
                'materials':ksec[2]/max(mean,1e-6),
                'logistics':ksec[5]/max(mean,1e-6)}
        if ARCHETYPES[p.project_id][0] in ('Grid','PowerGen'):
            ratios['energy']+=.30 # construction of supply itself does not require completed additional supply
        if ARCHETYPES[p.project_id][0] in ('Logistics','MachineTools'):
            ratios['logistics']+=.15
        if min(ratios.values())<.62*difficulty:
            return False,'resource'
        jobs=np.bincount(e.job,minlength=32)
        if jobs[p.firm]<6 and ARCHETYPES[p.project_id][0] in ('SemiDesign','Robotics','IndustrialAI','AICloud'):
            return False,'skills'
        # Demand checks rely on OBSERVED past shortages, no future forecasts.
        # Long-lived infrastructure may be needed even at zero current shortage.
        if (p.project_id in (6,11,20)) and e.shortages:
            avg=np.mean(np.array(e.shortages[-min(6,len(e.shortages)):]),axis=0)
            if avg[ARCHETYPES[p.project_id][1]]<.003 and p.delayed<3:
                return False,'demand'
        return True,'ok'

    def commission(self,e,m):
        pending=[]
        for p in self.projects:
            if p.maturity>m:
                pending.append(p);continue
            allowed,reason=self._can_complete(e,p,m)
            if not allowed and p.delayed<self.completion_horizon:
                p.delayed+=1;self.gate_denials+=1;pending.append(p);continue
            if not allowed:
                # sunk unfinished capital, no hidden transfer into productive capacity
                self.cancelled+=1;self.written_off+=p.paid_capital;continue
            e.k[p.firm]+=p.paid_capital
            self.completed+=1;self.commissioned_capital+=p.paid_capital;self.delay_total+=p.delayed
        self.projects=pending

    def record(self,m, e):
        self.month_history.append(dict(month=m+1,mode=self.mode,projects_started=self.started,
         projects_completed=self.completed,projects_cancelled=self.cancelled,
         cumulative_wip_investment=self.capital_spent,
         completed_real_capital=self.commissioned_capital,
         written_off_real_capital=self.written_off,
         active_wip_capital=sum(p.paid_capital for p in self.projects),
         gate_denials=self.gate_denials,average_extra_delay=self.delay_total/max(self.completed,1)))

    def audit(self):
        wip=sum(p.paid_capital for p in self.projects)
        assert abs(self.capital_spent - self.commissioned_capital - self.written_off - wip)<1e-6
        assert wip>=-1e-10