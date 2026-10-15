"""Fail-closed adapter for verified 98-product Russian IO tables.
Input normalized CSV exported from Rosstat workbook after manual sheet verification.
NEVER label illustrative A as official. Units: consistent RUB, gross output in same prices/year.
"""
from __future__ import annotations
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
SECTORS=['agriculture','energy','materials','machinery','construction','transport','trade','services']
def load_rows(path):
 with open(path,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def build_map(mapping,intermediate,outputs,root_source_path,year):
 """mapping: product_code,sector; intermediate: input_code,output_code,intermediate_rub;
 outputs: product_code,gross_output_rub. Includes imports only if explicitly labeled.
 """
 ps=load_rows(mapping); inter=load_rows(intermediate); outs=load_rows(outputs)
 mp={z['product_code']:z['sector'] for z in ps}
 assert len(mp)==len(ps) and set(mp.values())<=set(SECTORS)
 byproduct={z['product_code']:float(z['gross_output_rub']) for z in outs}
 assert len(byproduct)==len(outs) and byproduct.keys()==mp.keys()
 secoutput=np.zeros(8)
 for code,val in byproduct.items():
  assert val>0 and np.isfinite(val)
  secoutput[SECTORS.index(mp[code])]+=val
 flow=np.zeros((8,8))
 for z in inter:
  i,j=z['input_code'],z['output_code']; v=float(z['intermediate_rub'])
  assert i in mp and j in mp and np.isfinite(v) and v>=0
  flow[SECTORS.index(mp[i]),SECTORS.index(mp[j])]+=v
 A=flow/secoutput[None,:]
 assert np.all(np.isfinite(A)) and np.all(A>=0) and np.max(A.sum(axis=0))<1.0,'matrix infeasible; verify prices/domestic/import and code mapping'
 src=Path(root_source_path)
 assert src.exists() and src.is_file(),'raw official Rosstat workbook is REQUIRED, not a guessed source'
 digest=hashlib.sha256(src.read_bytes()).hexdigest()
 return dict(year=int(year),sectors=SECTORS,direct_cost_matrix=A.tolist(),sector_gross_output_rub=secoutput.tolist(),source_original_filename=src.name,source_sha256=digest,intermediate_rows=len(inter),product_count=len(mp),status='NORMALIZED_FROM_PROVIDED_SOURCE_NOT_AUTOMATICALLY_VERIFIED_AS_OFFICIAL')
def main():
 p=argparse.ArgumentParser();p.add_argument('--map',required=True);p.add_argument('--intermediate',required=True);p.add_argument('--outputs',required=True);p.add_argument('--original-rosstat-workbook',required=True);p.add_argument('--year',type=int,required=True);p.add_argument('--out',required=True)
 a=p.parse_args();v=build_map(a.map,a.intermediate,a.outputs,a.original_rosstat_workbook,a.year)
 Path(a.out).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
 print('Validated normalized coefficient matrix; provenance SHA256:',v['source_sha256'])
if __name__=='__main__':main()