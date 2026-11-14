#!/usr/bin/env python3
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();spec={'v11':'fqie_contract_output_v11_20261113','v12':'fqie_contract_output_v12_20261114','v13':'fqie_contract_output_v13_20261115'};rows=[]
 for tag,d in spec.items():
  h=a.root/'target_aware_counting'/'output'/d;cm=json.loads((h/'fair_count_metrics.json').read_text());bs=json.loads((h/'fair_bootstrap.json').read_text());sg=json.loads((h/'fair_sign_test.json').read_text());sa=json.loads((h/'fair_absent_safety.json').read_text()) if (h/'fair_absent_safety.json').exists() else None
  ci=bs['all']['bootstrap95'];rows.append({'cohort':tag,'delta_mae':bs['all']['delta_mae'],'bootstrap95':ci,'ci_excludes_zero':ci[1]<0,'sign_p':sg['all']['p_two_sided'],'sign_pass':sg['all']['p_two_sided']<.05,'absent_rate':sa['safe_iic']['rate'] if sa else None,'baseline_absent_rate':sa['baseline']['rate'] if sa else None,'eligible':ci[1]<0 and sg['all']['p_two_sided']<.05 and (sa is None or sa['safe_iic']['rate']<=sa['baseline']['rate'])})
 eligible=[x['cohort'] for x in rows if x['eligible']];out={'status':'FQIE_FAIR_CLAIM_GATE','policy':'preserve_baseline','minimum_independent_cohorts':2,'eligible_cohorts':eligible,'gate_pass':len(eligible)>=2,'cohort_checks':rows,'allowed_claim':'FQIE is a conservative GT-blind rescue protocol with evidence-aware abstention; strong improvement claims remain locked until independent fair replications pass.'}
 a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
