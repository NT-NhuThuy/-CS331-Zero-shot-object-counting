# Import this reorganization safely into the original GitHub repository

This package was prepared **locally** and is not pushed to GitHub. Do not
force-push, delete historical branches, or overwrite `main` without review.

## What you receive

- `CS331_ASTAR_REORGANIZED_ALL_BRANCHES.bundle`: complete local Git history,
  including original refs plus `research/a-star-unification`.
- `CS331_ASTAR_REORGANIZED_WORKTREE.zip`: readable files of the proposed new
  branch; this ZIP does NOT contain `.git` history.
- `CS331_ASTAR_RESEARCH_REPORT.md`: standalone copy of the main report.

## Import into an existing local clone

Put the downloaded `.bundle` alongside your repo, then:

```bash
git fetch ../CS331_ASTAR_REORGANIZED_ALL_BRANCHES.bundle \
  refs/heads/research/a-star-unification:refs/heads/research/a-star-unification
git switch research/a-star-unification
python -m pip install -e '.[test]'
python -m pytest -q tests legacy/thuy/branch legacy/loc/tools/praco_test1/test_verify_praco_artifacts.py
# after inspecting changes and getting team approval:
git push origin research/a-star-unification
```

On Windows PowerShell, put `git fetch ...` on one line (without Unix `\`
line continuation). The existing `main` and all other branches remain intact.
Create a GitHub Pull Request from `research/a-star-unification` to `main` only
after code-owner review. This restructuring intentionally archives old paths;
notebooks should be executed from the new `legacy/<owner>` directory, not
with obsolete root-level `Lộc/...` paths.

## What's actually been done

- 530 original main tracked files preserved by Git moves; all content attributed
  through history.
- Full V7.6 official PrACo adapter additions from Lộc's separate branch
  imported into `legacy/loc` (with Git author lineage still discoverable).
- New unified protocol tooling and CPU CI added, NOT a false fully integrated
  CountGD++ inference program.
- Individual historical experiment results remain immutable descriptive data.

## No accidental scientific claims

The new branch is for **research infrastructure and method selection**. It
must not be cited as an empirical paper artifact proving a new fused method;
the previously observed holdouts/test sets are not reset by renaming paths.
