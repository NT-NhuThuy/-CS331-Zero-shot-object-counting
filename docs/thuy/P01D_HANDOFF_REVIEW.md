# P01-D — Review before integration

**Recommended PR base:** `research/unified-integration-verified`

## Evidence validation

- Original bundle has three files, and its archive passed an integrity check.
- Frozen manifest has 13 experiment inventory records (E00–E10, V13, V14).
- V14 has 54 distinct query IDs and 54 entries missing source provenance.
- Zero verified source image SHA256 values; no source group may be inferred.
- Paths in manifest are workspace-relative; it embeds no GT, image bytes,
  checkpoints or obvious credentials.
- The single absolute server path in the document has been replaced with
  a portable placeholder; the manifest is byte-for-byte unchanged.

## Blockers / limitations

- Auditor does not actually search E00–E10 / V13 roots: `NOT_LOCATED` is a
  supplied scoped-inventory claim, **not an independently replayed result**.
  Request search scope/logs or raw inventory evidence from the original owner.
- V14 source image paths/hashes and task-index bytes are not recovered.
- P01-D inventory-only PASS does not unblock G0. No independent split claim.
- Running the auditor requires the original private V14 artifacts; repository
  publication alone is not sufficient to reproduce 54 records.

## Hygiene

- Never commit any file under `UNICOUNT_PRIVATE/`, image bytes, keys, or GT.
- The included manifest is only lineage metadata; treat it as historical
  `UNKNOWN` provenance for P03 cross-checks.
- Added tests use synthetic fixtures and do not access private datasets.
