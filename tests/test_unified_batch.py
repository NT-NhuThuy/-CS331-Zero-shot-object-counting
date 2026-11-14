import json
import subprocess
import sys
from pathlib import Path


def test_batch_cli_on_frozen_fixture(tmp_path):
    root = Path(__file__).resolve().parents[1]
    inp = tmp_path / "inputs"
    out = tmp_path / "outputs"
    inp.mkdir()
    (inp / "toy.json").write_bytes((root / "fixtures" / "unified_toy.json").read_bytes())
    run = subprocess.run([sys.executable, "-m", "countlab.method.batch", "--inputs", str(inp),
                          "--outputs", str(out), "--config", str(root / "configs/method/unicount_v1.json")],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    result = json.loads((out / "toy.json").read_text())
    assert result["count"] == 2
    run2 = subprocess.run([sys.executable, "-m", "countlab.method.batch", "--inputs", str(inp),
                           "--outputs", str(out), "--config", str(root / "configs/method/unicount_v1.json")],
                          capture_output=True, text=True)
    assert run2.returncode != 0 and "FileExistsError" in run2.stderr
