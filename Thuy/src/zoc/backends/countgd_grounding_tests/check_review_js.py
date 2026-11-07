"""Optional JS checks: python tests/check_review_js.py (Node.js required)."""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from cgground.review import HTML

if not shutil.which('node'):
    raise SystemExit('Node.js is required only for these optional JavaScript checks.')
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'review.js'
    path.write_text(re.findall(r'<script>(.*?)</script>',HTML,re.S)[0],encoding='utf-8')
    subprocess.run(['node','--check',str(path)],check=True)
    subprocess.run(['node',str(Path(__file__).with_name('test_review_js.cjs')),str(path)],check=True)
