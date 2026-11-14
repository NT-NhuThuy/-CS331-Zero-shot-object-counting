# Source-artifact audit

Audit date: 2026-11-07. This is a provenance audit of the selected archive, not a rerun of GPU inference.

## Audited source references

| Experiment | Archived source evidence | Audit status |
|---|---|---|
| E00 | `reports/source_reports/E00/baseline_source_record.md` | summarized source record; full handoff and runtime artifacts not archived |
| E01 | `reports/source_reports/E01/pseudo_exemplar_source_record.md` | summarized source record; multiple protocol runs remain separate |
| E02 | `reports/source_reports/E02/candidate_recall_source_record.md` | summarized diagnostic; denominator remains audit-specific |
| E03 | `reports/source_reports/E03/a4_source_record.md` | summarized VAL300 source-disjoint result; not comparable to E00 primary100 as a paired claim |
| E04 | `reports/source_reports/E04/b0_source_record.md` | summarized primary100 comparison; absent diagnostic remains separate |
| E05 | `reports/source_reports/E05/multiview_source_record.md` | summarized negative diagnostic |
| E06 | working notes and semantic probe reports | diagnostic; source report to be added if hard-filter run is released |
| E07 | `reports/source_reports/E07/box_merging_source_record.md` | summarized negative diagnostic |
| E08 | `reports/source_reports/E08/sam2_source_record.md` | summarized mask/relationship diagnostic |
| E09 | input-intervention report family | reported trade-off; no pooled headline result |
| E10 | pseudo-exemplar reports and handoff notes | protocol-sensitive; no pooled headline result |

## Important limitations

- No inference was rerun during this archive build.
- Checkpoint binaries and datasets are intentionally excluded.
- The original absolute paths, exact runtime environment, and all per-run hashes are not yet portable to a public repository.
- Therefore the experiment records remain `reported_pending_audit` until a future release adds complete manifests and public download instructions.

## Scientific interpretation

The archived evidence supports component-level claims for frozen baseline characterization, A4 candidate recovery, and B0 SAM3 add-only recovery. It does not establish a final query-safe counting method.
