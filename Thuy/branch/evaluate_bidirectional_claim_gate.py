#!/usr/bin/env python3
"""Claim gate for frozen development and locked source-disjoint holdout."""
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--development',type=Path,required=True);p.add_argument('--holdout',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 dm=json.loads((a.development/'offline_metrics.json').read_text());dp=json.loads((a.development/'paired_audit.json').read_text());hm=json.loads((a.holdout/'offline_metrics.json').read_text());hp=json.loads((a.holdout/'paired_audit.json').read_text())
 checks={'development_ci_excludes_zero':dp['bootstrap95'][1]<0,'holdout_ci_excludes_zero':hp['bootstrap95'][1]<0,'holdout_significant':hp['sign_test']['p_two_sided']<.05,'holdout_absent_nonworse':hm['absent_bidirectional']['mae']<=hm['absent_baseline']['mae'],'holdout_wins_exceed_losses':hp['sign_test']['wins']>hp['sign_test']['losses']}
 passed=all(checks.values());out={'status':'FQIE_BIDIRECTIONAL_CLAIM_GATE','gate_pass':passed,'checks':checks,'development':{'baseline_mae':dm['baseline']['mae'],'method_mae':dm['bidirectional']['mae'],'bootstrap95':dp['bootstrap95']},'locked_holdout':{'baseline_mae':hm['baseline']['mae'],'method_mae':hm['bidirectional']['mae'],'bootstrap95':hp['bootstrap95'],'sign_p':hp['sign_test']['p_two_sided'],'absent_baseline_mae':hm['absent_baseline']['mae'],'absent_method_mae':hm['absent_bidirectional']['mae']},'allowed_claim':'On the locked source-disjoint LVIS-derived holdout, frozen bidirectional evidence correction significantly reduces CountGD++ MAE without worsening absent-query error.' if passed else 'Strong empirical claims remain locked.'}
 a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
