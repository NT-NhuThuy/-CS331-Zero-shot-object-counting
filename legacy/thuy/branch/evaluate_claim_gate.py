#!/usr/bin/env python3
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();x=json.loads(a.report.read_text());rows=[]
 for tag,v in x['cohorts'].items():
  ci=v['bootstrap']['all']['bootstrap95'];present=v['sign_test']['present'];safe=v['absent_safety']['safe_iic'];base=v['absent_safety']['baseline']
  rows.append({'cohort':tag,'ci_excludes_zero':ci[1]<0,'present_sign_p_lt_005':present['p_two_sided']<.05,'absent_rate_not_higher':safe['rate']<=base['rate'],'eligible':ci[1]<0 and present['p_two_sided']<.05 and safe['rate']<=base['rate']})
 eligible=[r['cohort'] for r in rows if r['eligible']]
 out={'status':'FQIE_CLAIM_GATE_AUDIT','minimum_independent_cohorts':2,'eligible_cohorts':eligible,'level_1_gate_pass':len(eligible)>=2,'cohort_checks':rows,'claim':'GT-blind proposal recovery with whole-instance identity control improves frozen CountGD++ on independent cohorts while not increasing absent false promotion.'}
 a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
