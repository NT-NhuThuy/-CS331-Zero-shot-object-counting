#!/usr/bin/env python3
"""Create a fresh, source-disjoint FQIE replication scaffold without copying outputs."""
from __future__ import annotations
import argparse, shutil
from pathlib import Path

ALLOW=('build_consensus.py','common.py','download_pool.py','identity_cycle.py',
       'prepare_full_p0.py','prepare_tile_job.py','prepare_zero_tiles.py',
       'prepare_pool.py','RUN_DOWNLOAD_POOL.sh','RUN_FULL_P0_GPU.sh',
       'RUN_IDENTITY_CYCLE_GPU.sh','RUN_ZERO_TILES_GPU.sh')
def main():
 p=argparse.ArgumentParser();p.add_argument('--template',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--tag',default='V12');p.add_argument('--seed',default='fqie-safe-iic-replication-v12');p.add_argument('--exclude',action='append',default=[]);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 a.out.mkdir(parents=True);old=a.template.name;new=a.out.name
 for name in ALLOW:
  src=a.template/name;dst=a.out/name
  if not src.is_file():raise FileNotFoundError(src)
  text=src.read_text().replace(old,new).replace('V11',a.tag).replace('V12',a.tag).replace('V13',a.tag).replace('fqie-safe-iic-quality-v11',a.seed).replace('fqie-safe-iic-replication-v12',a.seed).replace('fqie-safe-iic-replication-v13',a.seed)
  if name=='prepare_pool.py':
   extra=','.join("ROOT/'target_aware_counting/output/"+x+"/pool/inference_coco.json'" for x in a.exclude)
   marker="ROOT/'target_aware_counting/output/fqie_safe_iic_validation_20261112_v10/pool/inference_coco.json'"
   if marker not in text:raise RuntimeError('prepare_pool exclusion marker missing')
   text=text.replace(marker,marker+(','+extra if extra else ''))
  dst.write_text(text);shutil.copymode(src,dst)
 excluded=', '.join(a.exclude) if a.exclude else 'none'
 (a.out/'PREREGISTER.md').write_text(f'''# FQIE {a.tag} locked holdout preregistration\n\n- Protocol: frozen bidirectional controller; reject only semantic=NO and whole=NO; UNKNOWN retains baseline; unmatched accepted rescue adds one; match IoU=0.5.\n- Seed: `{a.seed}`.\n- Cohort: 90 present + 90 verified-absent, source-disjoint from: `{excluded}`.\n- Inference: frozen CountGD++, `num_exemplars=0`, `num_pseudo_exemplars=3`, GT-blind.\n- Primary endpoints: all-query/present MAE, absent false-promotion, paired bootstrap CI, exact paired sign test.\n- This cohort is opened once and is not used to tune the method.\n''')
 print({'status':'REPLICATION_SCAFFOLD_CREATED','out':str(a.out),'files':len(ALLOW)+1,'source_exclusion':'V7,V8,V9,V10,V11'})
if __name__=='__main__':main()
