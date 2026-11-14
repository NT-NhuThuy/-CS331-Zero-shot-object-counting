# E03 - A4 Crop + NMS fallback

## Objective

Test whether crop-based candidate recovery improves missed-instance recall.

## Setup

VAL300, source-disjoint cohort. This is not directly comparable to E00 primary100.

## Result

Reported MAE 17.07 → 14.95, TP +620, FP +21. Pending audit of manifest, evaluator, and source logs.

## Decision

`KEEP` — candidate recovery evidence.
