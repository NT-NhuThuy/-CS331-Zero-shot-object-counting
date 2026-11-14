#!/usr/bin/env bash
# Run on the original WSL laptop if you want to add the EXACT notebook A source.
set -euo pipefail
repo_root="${1:-.}"
dest="${repo_root}/Lộc/notebooks/06_praco_official_adapter"
mkdir -p "$dest"
notebook="${2:-}"
if [[ -z "$notebook" ]]; then
  echo 'Find the original notebook first:'
  echo '  find "$HOME/PrACo_Local" /mnt/c/Users -type f -name "CS331_PrACo_RTX5060_WSL2_FULL_Local.ipynb" 2>/dev/null'
  echo 'Usage: bash collect_original_a_notebook.sh <repo-root> <full-path-to-original-notebook>'
  exit 2
fi
if [[ ! -f "$notebook" ]]; then echo "Not found: $notebook" >&2; exit 1; fi
python3 - "$notebook" "$dest/CS331_PrACo_RTX5060_WSL2_FULL_Local.ipynb" <<'PY'
import json,sys
from pathlib import Path
src=Path(sys.argv[1]); dst=Path(sys.argv[2])
nb=json.loads(src.read_text(encoding='utf-8'))
assert nb['nbformat']==4
for cell in nb['cells']:
    if cell['cell_type']=='code':
        cell['outputs']=[]; cell['execution_count']=None
    # Strip cell metadata potentially containing private mounted paths/tokens.
    cell['metadata']={k:v for k,v in cell.get('metadata',{}).items() if k in ('tags','collapsed')}
nb['metadata']={k:v for k,v in nb.get('metadata',{}).items() if k in ('kernelspec','language_info')}
dst.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print('Copied original A notebook source (outputs cleared):',dst)
PY
