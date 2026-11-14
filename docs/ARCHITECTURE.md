# Current architecture (working method v1)

The current unified method is specified in [`method/METHOD_V1.md`](method/METHOD_V1.md) and [`method/INTERFACE_CONTRACT.md`](method/INTERFACE_CONTRACT.md). The old proposal-only architecture is superseded by these documents.

- Frozen CountGD++ candidate output (requires future online adapter)
- Thư: risk router (deterministic CPU proof of contract; learned model not connected)
- Lộc: multi-view proposal source + instance-level unary verifier (only frozen replay now)
- Thúy: complete-link conservative identity + bidirectional reconciliation (CPU runnable)
- Structured partial/abstained output, evidence provenance and separate GT evaluation

**Status:** one real replay code path and unit tests, not an end-to-end GPU model or benchmark-proven new method. Original researcher implementations remain under `legacy/` unchanged.
