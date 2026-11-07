# Experimental protocol

1. Keep CountGD++ weights frozen unless an experiment explicitly changes the research scope.
2. Do not infer `absent` from missing annotations. Use `present`, `verified_absent`, or `unknown`.
3. Development/validation results must not be described as test results.
4. Never pool metrics across different cohorts, splits, or evaluator versions.
5. Record the dataset version, image/cohort IDs, checkpoint hash, source commit, command, seed, evaluation protocol, and metric file.
6. Report limitations and invalid executions explicitly.
