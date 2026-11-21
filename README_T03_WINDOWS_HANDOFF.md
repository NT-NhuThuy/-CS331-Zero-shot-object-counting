# T03 — Additive Windows-local handoff

This ZIP is an **uncommitted additive patch**, not a Git repository and not a server-side modification. It was tested against the user-supplied `UNICOUNT_V2_3fb58ccedb69.tar` snapshot, **not** the live Windows worktree or an assumed remote GitHub HEAD. Do **not** overwrite locally modified files. All ZIP entries are new paths relative to the repo root.

## Windows PowerShell sequence

Save this ZIP in `Downloads`, open PowerShell in your actual Windows checkout root (e.g. `D:\EXERCISE\CS331\UNICOUNT_THUY_V2`), and inspect before applying:

```powershell
cd D:\EXERCISE\CS331\UNICOUNT_THUY_V2
git status --short
git branch --show-current
$Zip = Join-Path $env:USERPROFILE 'Downloads\UNICOUNT_C3_T03_LOCAL_HANDOFF_20261121.zip'
$Archive = [System.IO.Compression.ZipFile]::OpenRead($Zip)
$Conflicts = @($Archive.Entries | Where-Object { $_.Name -and (Test-Path -LiteralPath (Join-Path (Get-Location) $_.FullName.Replace('/', '\'))) })
$Archive.Dispose()
if ($Conflicts.Count -gt 0) { $Conflicts.FullName; throw 'HOLD: T03 destination file already exists' }
Expand-Archive -LiteralPath $Zip -DestinationPath .
$env:PYTHONPATH = 'src'
py -m pytest -q tests/test_thuy_t03_contract_v1.py
py tools/thuy/t03_validate_v1.py --examples docs/method/T03_example_input_output.json
py -m pytest -q
git status --short
```

If `[System.IO.Compression.ZipFile]` is missing, first run `Add-Type -AssemblyName System.IO.Compression.FileSystem`. Do not push or commit until the tests pass on **your** checkout and you have reviewed `git diff --no-index` or staged changes. Your local checkout may contain newer code than the archived snapshot; if an output contract test fails because of newer solver semantics, compare the change with T03_OWNER and version the contract rather than silently weakening the test.

## Windows Git stage/commit, after local PASS only

```powershell
git add README_T03_WINDOWS_HANDOFF.md src/countlab/method/t03_contract_v1.py docs/method/evidence_graph_schema_v1.json docs/method/T03_OUTPUT_CONTRACT_V1.md docs/method/T03_example_input_output.json docs/method/T03_LOCAL_ACCEPTANCE_REPORT.md tests/test_thuy_t03_contract_v1.py tools/thuy/t03_validate_v1.py T03_HANDOFF_MANIFEST.json
git diff --cached --stat
git diff --cached --check
git commit -m "feat(c3-t03): add strict GT-blind evidence graph contract"
```

Only on Windows, send/push the commit to the appropriate current feature branch and open the assigned PR into `research/unified-integration-verified` after CI. Never execute Git/GitHub commands on the remote Linux server.

## What to send to teammates

`T03_HANDOFF_MANIFEST.json` plus `docs/method/T03_OUTPUT_CONTRACT_V1.md`, the schema and the audited test output. The strict validator is opt-in until input producer compatibility is signed off. T06/T09/T12 delivery and further research are separate tasks.
