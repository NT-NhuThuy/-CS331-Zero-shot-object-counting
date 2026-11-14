# Experiment registry / claim accounting

`configs/regimes` contains three **incompatible evaluation settings**. Do not
pool their metrics, count a trained correction head as a training-free model,
or compare different CountGD++ operating points without a matched rerun.

`paper/claim_ledger.csv` is the current descriptive evidence ledger. This
branch deliberately does not copy raw test labels into inference inputs.

For new experiments, create a protocol manifest BEFORE accessing target GT:

- `experiment_id`, owner, Git commit, dataset license, source IDs, split hash;
- model checkpoint SHA, tokenizer/model revision, pseudo exemplar count;
- prompts, crop scales, thresholds, candidate source, verifier budget;
- locked decision rule, UNKNOWN/abstention semantics, runtime hardware;
- model outputs sealed and hashed before using GT for scoring;
- validation/test separation and explicit prior exposure log.

Use `countlab-audit` for matched source-cluster metrics on a frozen prediction
CSV, and `countlab` contracts for schema checks. This tooling does NOT run any
of the GPU inference pipelines in `legacy/`.
