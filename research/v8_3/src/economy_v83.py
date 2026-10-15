"""V8.3: V8.2 monetary engine + real EIIOM-style capital WIP/project factory.
No synthetic benefit is applied to SYNERGY directly; see allocation rules.
"""
from dataclasses import dataclass
import numpy as np
from economy_v82 import Economy82,Config82
from economy_v81_legacy import MODES,POLICIES,A
from eurasian_factory import ProjectFactory

@dataclass
class Config83(Config82):
    enable_projects:bool=True
    w_shortage:float=.7

class Economy83(Economy82):
    def __init__(self,cfg:Config83):
        super().__init__(cfg)
        self.factory=ProjectFactory(cfg.mode,cfg.seed,A,cfg.w_shortage)
        self.project_mode=bool(cfg.enable_projects)

    def month_step(self,m):
        kprev=self.k.copy()
        if self.project_mode:
            self.factory.commission(self,m)
            kprev=self.k.copy()
        super().month_step(m)
        if self.project_mode:
            self.factory.collect_real_investment(self,kprev,m)
            self.factory.audit()
            self.factory.record(m,self)
            self.history[-1].update(self.factory.month_history[-1])
        else:
            self.history[-1].update(projects_started=0,projects_completed=0,
                projects_cancelled=0,cumulative_wip_investment=0,
                completed_real_capital=0,written_off_real_capital=0,active_wip_capital=0,
                gate_denials=0,average_extra_delay=0)
        self.audit()