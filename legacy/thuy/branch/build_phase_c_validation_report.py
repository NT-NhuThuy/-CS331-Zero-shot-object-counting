#!/usr/bin/env python3
"""Aggregate the completed CPU validation phase into one auditable report."""
import argparse,json,subprocess,sys
from pathlib import Path
def read(p):return json.loads(Path(p).read_text()) if Path(p).is_file() else {'status':'MISSING','path':str(p)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();h=a.experiment;root=Path(__file__).parent
 tests=subprocess.run([sys.executable,'-m','unittest','test_fqie_contracts.py','test_safe_iic.py','test_hard_case_annotation.py','test_hard_case_selection.py','test_annotation_templates.py','test_autofill_lvis_targets.py'],cwd=root,capture_output=True,text=True)
 ann=h/'hard_case_annotations_lvis_other_dev.jsonl';valid=subprocess.run([sys.executable,'validate_annotation_batch.py',str(ann.resolve())],cwd=root,capture_output=True,text=True)
 report={'status':'FQIE_PHASE_C_VALIDATION_CPU_COMPLETE','uses_gt_for_inference':False,'experiment':str(h),'tests':{'returncode':tests.returncode,'stdout':tests.stdout,'stderr':tests.stderr},'annotation_batch':{'returncode':valid.returncode,'stdout':valid.stdout,'stderr':valid.stderr},'artifacts':{'selection_lock':read(h/'hard_case_selection_lock.json'),'proposal_quality':read(h/'proposal_quality_dev_audit.json'),'instance_evidence':read(h/'instance_evidence_dev_metrics.json'),'safe_iic_output':read(h/'evidence_output_safe_iic_diagnostic.json'),'annotation_template_rows':sum(1 for z in ann.read_text().splitlines() if z.strip()) if ann.is_file() else 0}}
 a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'tests_returncode':tests.returncode,'annotation_returncode':valid.returncode,'out':str(a.out)},indent=2))
if __name__=='__main__':main()
