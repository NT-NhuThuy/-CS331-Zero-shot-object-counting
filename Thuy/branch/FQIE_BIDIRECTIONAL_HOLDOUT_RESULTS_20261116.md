# FQIE bidirectional evidence controller — locked holdout results

## Protocol

The method was developed on V13 and frozen before opening the source-disjoint V14 cohort (90 present and 90 verified-absent queries). CountGD++ remained frozen and inference did not read GT. A baseline proposal is removed only when semantic and whole-instance checks both return `NO`; `UNKNOWN` retains it. A verified rescue is added only when unmatched to a baseline proposal at IoU 0.5.

## Primary V14 holdout result

| Metric | Frozen CountGD++ | Bidirectional FQIE |
|---|---:|---:|
| All MAE | 1.7611 | **1.1000** |
| Present MAE | 3.3556 | **2.2000** |
| Absent MAE | 0.1667 | **0.0000** |
| Exact accuracy | 56.67% | **60.56%** |

Paired outcomes: 22 improvements, 3 regressions, 155 ties. Delta MAE was -0.6611; source-image clustered bootstrap 95% CI was [-1.3833, -0.1833] (10,000 resamples), and exact paired sign-test p=0.0001565.

## Post-freeze ablation

| Arm | All MAE | Present MAE | Absent MAE | Exact |
|---|---:|---:|---:|---:|
| Baseline | 1.7611 | 3.3556 | 0.1667 | 56.67% |
| Rescue only | 1.7444 | 3.3222 | 0.1667 | 57.22% |
| Rejection only | 1.1167 | 2.2333 | **0.0000** | 60.00% |
| Full bidirectional | **1.1000** | **2.2000** | **0.0000** | **60.56%** |

Evidence-based rejection is the dominant contribution; conservative rescue adds a smaller gain. All three regressions were present-query over-pruning failures. No rule was changed after V14 was opened. These results support a significant improvement claim on this locked LVIS-derived holdout, not universal/external-dataset/SOTA claims.

## Frozen artifact hashes

- `full_cohort_output.json`: `94e71b3f9668e122aefa698480189081284fe0095ee3a0b6c25bfe1b85a24fd9`
- `offline_metrics.json`: `e553a87bc31a6c265f7b5e988ac4e1a6769305aa6bc727a85d16b6074598ef06`
- `paired_audit.json`: `9d250bdcdcb75e53452d3db6a6ad30cb5456594e5cee5e4bd6f9e1d43779f696`
- `ablation_metrics.json`: `39d437ca5834305dea0bdc80bcdec2b6f86936a4cc938aa11358821052f1ba27`
- `failure_audit.json`: `4fd8fe2944ec19fc9bb8e8380e8f0fba6e223610ef6dabf16eff1353c0c2b355`
