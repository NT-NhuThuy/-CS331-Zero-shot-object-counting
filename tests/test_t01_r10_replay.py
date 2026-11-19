import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rawreplay',ROOT/'tools/t01/audit_r10_raw_cpu_replay.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_real_r10_reconstructs_all_raw_and_parser():
    result=module.audit(ROOT/'docs/method/parity_logs/UNICOUNT_T01_R10_GPU_EVIDENCE.zip',
                        ROOT/'docs/method/parity_logs/r10_image_dimensions_verified.json')
    assert result['status']=='RAW_CPU_PARSER_REPLAY_PASS_NOT_INDEPENDENT_PARITY'
    assert result['pairs']==32
    assert result['candidates']==38
    assert result['stage_raw_sha_checked']==64
    assert result['g0']=='HOLD'
