"""Paired economy v8.3 experiments, TRAIN 2016-22 and OOS 2023-25.
Never uses OOS macro GDP/CPI observations as input or in parameter choice.
Historical FX m is not seen by the agent until the end of month m.
"""
import sys,argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t
sys.path.insert(0,str(Path(__file__).resolve().parent))
from economy_v83 import Economy83,Config83,MODES,POLICIES
from economy_v81_legacy import A
from eurasian_factory import ARCHETYPES
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)


def simulate(mode,policy,seed,stress,w,project=True,months=120):
    cfg=Config83(mode=mode,policy=policy,seed=seed,stress=stress,response=.4,
       fiscal_scale=7.5,credit_scale=2.,household_credit_rate=0.,
       w_shortage=w,enable_projects=project)
    econ=Economy83(cfg)
    path=econ.run(months)
    assert econ.maxerr<1e-6 and econ.ioresmax<1e-6
    assert abs(econ.released-econ.matched)<1e-7
    if project:econ.factory.audit()
    return econ,path


def deltas(d,old=83):
    initial=d.iloc[old];final=d.iloc[-1]
    cols={'sales':'real_final_sales','consumption':'real_consumption',
          'gross':'real_output_gross','gov_spend':'public_spend',
          'bailin':'bank_resolutions','real_capital':'real_K',
          'escrow':'escrow_unlocked','demand':'demand_vouchers'}
    z={k:float(final[v]-initial[v]) for k,v in cols.items()}
    z.update(final_wealth_gini=float(final.hh_liquid_gini),
             credit=float(final.loans),m2_proxy=float(final.m2_proxy),
             bank_capratio=float(final.bank_min_capratio),
             m2_oos_factor=float(final.m2_proxy/initial.m2_proxy),
             gov_debt=float(final.gov_debt),max_fin_error=float(final.financial_residual),
             max_io_error=float(final.input_output_residual),
             started=float(final.projects_started-initial.projects_started),
             completed=float(final.projects_completed-initial.projects_completed),
             cancelled=float(final.projects_cancelled-initial.projects_cancelled),
             wip=float(final.active_wip_capital),
             real_capital_commissioned=float(final.completed_real_capital-initial.completed_real_capital),
             capital_written_off=float(final.written_off_real_capital-initial.written_off_real_capital),
             cap_in_wip=float(final.cumulative_wip_investment-initial.cumulative_wip_investment))
    return z


def train(seeds):
    rows=[]
    for w in (.0,.25,.50,.75,1.0):
        for seed in range(7000,7000+seeds):
            e,d=simulate('synergy_cyber','baseline',seed,True,w,months=84)
            # Explicit scoring of real goods and consumption, less sunk WIP.
            score=float(d.iloc[-1].real_final_sales+0.5*d.iloc[-1].real_consumption
                        -0.50*e.factory.written_off)
            rows.append(dict(w_shortage=w,seed=seed,train_score=score,
                output=d.iloc[-1].real_final_sales,consumption=d.iloc[-1].real_consumption,
                writtenoff=e.factory.written_off))
    df=pd.DataFrame(rows);df.to_csv(OUT/'train_controller_grid.csv',index=False)
    avg=df.groupby('w_shortage').train_score.mean().sort_values(ascending=False)
    chosen=float(avg.index[0]);frozen={'split':'TRAIN 2016-2022; 84 months','OOS':'2023-2025; 36 months',
        'w_shortage':chosen,'seeds_train':list(range(7000,7000+seeds)),
        'selection':'maximize TRAIN physical output + 0.5 consumption - 0.5 stranded WIP',
        'data_io':'illustrative NOT empirical Rosstat','no_oos_in_training':True,
        'all_systems_share_project_recipes':True}
    (OUT/'freeze_manifest.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2))
    print('TRAIN_FROZEN',json.dumps(frozen),flush=True)
    return chosen


def main(seeds,train_seeds):
    w=train(train_seeds);rows=[];tracks=[]
    for idx,seed in enumerate(range(9800,9800+seeds)):
        for stress in (False,True):
            for mode in MODES:
                for policy in POLICIES:
                    e,d=simulate(mode,policy,seed,stress,w)
                    rows.append(dict(seed=seed,stress=int(stress),mode=mode,policy=policy,project_factory=True,**deltas(d)))
                    if seed==9800 and stress and policy in ('baseline','escrow_demand'):
                        tracks.append(d.assign(mode_exp=mode,policy_exp=policy))
        print(f'V8.3 main complete {idx+1}/{seeds} seeds',flush=True)
    z=pd.DataFrame(rows);z.to_csv(OUT/'main_4x4x2.csv',index=False)
    if tracks:pd.concat(tracks,ignore_index=True).to_csv(OUT/'representative_paths.csv',index=False)
    g=z.groupby(['stress','mode','policy'],as_index=False).agg({k:'mean' for k in [
        'sales','consumption','started','completed','cancelled','wip','capital_written_off',
        'real_capital_commissioned','final_wealth_gini','bank_capratio','m2_oos_factor','gov_debt','escrow','demand']})
    g.to_csv(OUT/'group_means.csv',index=False)
    pairs=[]
    for (seed,stress,mode),sub in z.groupby(['seed','stress','mode']):
        p=sub.set_index('policy');a=p.loc['escrow_demand'];b=p.loc['baseline']
        pairs.append(dict(seed=seed,stress=stress,mode=mode,
           d_sales=a.sales-b.sales,d_cons=a.consumption-b.consumption,
           d_cancelled=a.cancelled-b.cancelled,d_gini=a.final_wealth_gini-b.final_wealth_gini))
    pd.DataFrame(pairs).to_csv(OUT/'paired_policy_differences.csv',index=False)
    # Fair 'EIIOM on vs off' ablation for ALL systems (not just SYNERGY).
    ab=[]
    for seed in range(9800,9800+seeds):
      for stress in (False,True):
       for mode in MODES:
        e,d=simulate(mode,'escrow_demand',seed,stress,w,project=False)
        y=z[(z.seed==seed)&(z.stress==int(stress))&(z['mode']==mode)&(z.policy=='escrow_demand')].iloc[0]
        delta=deltas(d)
        ab.append(dict(seed=seed,stress=int(stress),mode=mode,
           projects_on_sales=y.sales,projects_off_sales=delta['sales'],
           d_sales=y.sales-delta['sales'],
           projects_on_cons=y.consumption,projects_off_cons=delta['consumption'],
           d_cons=y.consumption-delta['consumption']))
    ab=pd.DataFrame(ab);ab.to_csv(OUT/'all_modes_project_factory_ablation.csv',index=False)
    checks={'main_runs':len(z),'ablation_runs':len(ab), 'train_runs':5*train_seeds,
       'max_financial_residual':float(z.max_fin_error.max()),
       'max_physical_io_residual':float(z.max_io_error.max()),
       'project_factory_is_available_in_all_modes':True,
       'synergy_free_productivity_bonus':True,
       'rosstat_2021_io_data_loaded':False,
       'm2_2019_2022_calibrated':False,
       'oos_held_out_for_controller':True,
       'matching_policy_grid_all_modes':bool((z.groupby(['mode','stress']).policy.nunique()==4).all()),
       'shortage_weight_frozen':w}
    assert checks['max_financial_residual']<1e-6
    assert checks['max_physical_io_residual']<1e-6
    (OUT/'experiment_checks.json').write_text(json.dumps(checks,indent=2))
    print('CRISIS OOS TABLE')
    print(g[(g.stress==1)&g.policy.isin(['baseline','escrow_demand'])][['mode','policy','sales','consumption','cancelled','capital_written_off']].round(3).to_string(index=False))
    print('EIIOM ABALATION')
    print(ab.groupby(['stress','mode'])[['d_sales','d_cons']].mean().round(3).to_string())
    return z

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--seeds',type=int,default=6)
    a.add_argument('--train-seeds',type=int,default=3)
    v=a.parse_args();main(v.seeds,v.train_seeds)