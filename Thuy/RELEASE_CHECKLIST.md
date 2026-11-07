# Release checklist

- [x] Research structure created.
- [x] Selected source code copied without datasets/checkpoints/runtime outputs.
- [x] E00–E10 experiment records created.
- [x] Archive validator added.
- [ ] Audit every numeric result against source logs and manifests.
- [ ] Record source commit and checkpoint SHA256 for every executable result.
- [ ] Complete third-party license and attribution table.
- [ ] Run `bash scripts/run_tests.sh` in a clean environment.
- [ ] Review secrets and large files with `git diff --stat` and `git status`.
- [ ] Initialize Git, create the first commit, and push only after review.
