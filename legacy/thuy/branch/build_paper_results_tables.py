#!/usr/bin/env python3
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=[]
 for tag,d in [('V11','fqie_contract_output_v11_20261113'),('V12','fqie_contract_output_v12_20261114')]:
  h=a.root/'target_aware_counting'/'output'/d;cm=json.loads((h/'fair_count_metrics.json').read_text());ro=json.loads((h/'rescue_only_metrics.json').read_text());bs=json.loads((h/'fair_bootstrap.json').read_text());sa=json.loads((h/'fair_absent_safety.json').read_text());rows.append((tag,cm,ro,bs,sa))
 lines=['# FQIE fair results tables (2026-11-13)','', 'All values use preserve-baseline fallback for queries without rescue evidence.','', '| Cohort | Full baseline MAE | Full FQIE MAE | Delta | Rescue queries | Rescue baseline MAE | Rescue FQIE MAE | Absent FP baseline→FQIE |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for tag,cm,ro,bs,sa in rows:lines.append(f"| {tag} | {cm['baseline']['mae']:.4f} | {cm['safe_iic']['mae']:.4f} | {cm['safe_iic']['mae']-cm['baseline']['mae']:.4f} | {ro['rescue_queries']} | {ro['baseline']['mae']:.4f} | {ro['rescue']['mae']:.4f} | {sa['baseline']['false_promotion']} → {sa['safe_iic']['false_promotion']} |")
 lines += ['', '## Interpretation', '', '- V11 shows a small fair improvement concentrated in four rescue changes; its exact sign test is not significant.', '- V12 is a near-null replication and increases absent false promotion under fair fallback.', '- The defensible contribution is therefore the evidence-grounded protocol and auditability, not a universal MAE superiority claim.', '', '## Statistical guardrail', '', '| Cohort | Paired bootstrap 95% CI for fair MAE delta | Exact sign-test p |', '|---|---:|---:|']
 for tag,cm,ro,bs,sa in rows:lines.append(f"| {tag} | [{bs['all']['bootstrap95'][0]:.4f}, {bs['all']['bootstrap95'][1]:.4f}] | {json.loads((a.root/'target_aware_counting'/'output'/({'V11':'fqie_contract_output_v11_20261113','V12':'fqie_contract_output_v12_20261114'}[tag])/'fair_sign_test.json').read_text())['all']['p_two_sided']:.4f} |")
 a.out.write_text('\n'.join(lines)+'\n');print({'status':'PAPER_RESULTS_TABLES_BUILT','out':str(a.out)})
if __name__=='__main__':main()
