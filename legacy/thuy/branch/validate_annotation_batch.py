#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from hard_case_annotation import validate_case
def main():
 p=argparse.ArgumentParser();p.add_argument('path',type=Path);a=p.parse_args();bad=[];n=0
 for i,line in enumerate(a.path.read_text().splitlines(),1):
  if not line.strip():continue
  n+=1
  try:x=json.loads(line)
  except Exception as e:bad.append({'line':i,'reason':f'json:{e}'});continue
  ok,reason=validate_case(x)
  if not ok:bad.append({'line':i,'query_id':x.get('query_id'),'reason':reason})
 print(json.dumps({'rows':n,'valid':n-len(bad),'invalid':len(bad),'errors':bad},indent=2));raise SystemExit(1 if bad else 0)
if __name__=='__main__':main()
