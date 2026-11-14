"""Explicit status registry. A registry entry is NOT a runnable fused method."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Method:
    key: str
    owner: str
    origin: str
    task: str
    scope: str
    integration: str
    training: str


METHODS = (
    Method("fqie_bidirectional", "Thuy", "legacy/thuy/branch/bidirectional_evidence_controller.py",
           "Baseline rejection + unmatched proposal admission", "LVIS-derived V13/V14 reported", "PRIMARY_CANDIDATE_UNVERIFIED_END_TO_END", "frozen backbone"),
    Method("fqie_safe_iic", "Thuy", "legacy/thuy/branch/safe_iic.py",
           "Instance identity consistency", "CPU unit tests", "VALIDATION_REQUIRED", "training-free controller"),
    Method("fqie_proposal_recovery", "Thuy", "legacy/thuy/branch/prepare_quality_identity_manifest.py",
           "Multi-view candidate recovery", "LVIS research", "OPTIONAL_RECOVERY", "frozen backbone"),
    Method("joint_v32", "Thu", "legacy/thu/scripts/p14_08b_09_joint_unified.py",
           "Learned HOW-MANY risk correction", "FSC-147 test+hard-negative exploratory", "LEARNED_BASELINE", "trained correction models"),
    Method("a3_semantic", "Thu", "legacy/thu/scripts/train_s1_semantic_verifier_a3_clip_v1.py",
           "Query presence/semantic gating", "FSC-147 test+hard-negative exploratory", "SEMANTIC_BASELINE", "trained head"),
    Method("qwen_dual_evidence_v7", "Loc", "legacy/loc/notebooks/01_original_pipeline/V7_Dual_Evidence_Router.ipynb",
           "Image-level Qwen verification and routing", "FSC-147 observed test", "VERIFIER_BASELINE", "frozen Qwen"),
    Method("multiscale_v75", "Loc", "legacy/loc/notebooks/04_validation/",
           "Multi-resolution rescue", "2 x 240 positive validations", "RECOVERY_BASELINE", "frozen models"),
    Method("praco_v76", "Loc", "legacy/loc/docs/PRACO_TEST1_A_B_RESULTS.md",
           "Official PrACo Test1 adapter and sealed A/B comparison", "34,510 pairs; observed TEST", "REQUIRED_BENCHMARK_BASELINE", "frozen Qwen"),
)


def by_key(name: str) -> Method:
    return next(m for m in METHODS if m.key == name)
