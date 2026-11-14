# FQIE literature and positioning audit

Date: 2026-11-13  
Scope: query-only/open-world counting and evidence-grounded instance verification.

## Positioning rule

FQIE must not claim novelty for adaptive tiling, grounding, semantic filtering, exemplar discovery, or identity matching in isolation. The defensible contribution is the protocol-level combination:

> frozen open-world counter → GT-blind zero/low-evidence recovery → semantic and whole-instance admission → conservative identity association → auditable verified-instance output with UNKNOWN/abstention.

## Comparison matrix

| Work | Main input/setting | Proposal or exemplar source | Whole physical-instance gate | Identity/duplicate control | UNKNOWN/abstention | Evidence-trace / verified set |
|---|---|---|---|---|---|---|
| CountGD / CountGD++ | open-world text, visual, and generalized prompts | CountGD detector; CountGD++ also supports pseudo-exemplars and richer positive/negative prompting | not the stated output contract | not the FQIE Safe-IIC contract | not the stated output contract | scalar count / detection output, not FQIE evidence contract |
| Zero-Shot Object Counting (CVPR 2023) | class-name-only ZSC | learned zero-shot counting model | not the FQIE gate | not the FQIE association contract | not the stated output contract | count metrics |
| Point, Segment and Count (CVPR 2024) | class names or exemplar boxes | detector + segmentation proposals | segmentation/localization, but not FQIE semantic/whole/identity state machine | not the FQIE conservative graph rule | not the stated output contract | point/segment/count outputs |
| CountSE (ICCV 2025) | text-guided open-set counting | soft exemplars selected and filtered at multiple scales | exemplar quality filtering, not whole physical-instance admission | clustering/filtering, not FQIE identity evidence | not the stated output contract | count metrics |
| SAM3Count | text-prompted open-vocabulary image/video counting | SAM3 concept masks | segmentation-based, not FQIE whole-instance state | video/object ID concerns are distinct from FQIE query-level Safe-IIC | not the stated output contract | count/mask/video outputs |
| VLCounting | language-driven exemplar grounding | language-guided exemplar grounding | not the FQIE admission contract | not the FQIE graph contract | not the stated output contract | count metrics |
| GroundCount | VLM counting with detector grounding | external/object detector grounding | grounding is used to reduce hallucination; no FQIE evidence-state contract | not the FQIE identity policy | not the stated output contract | count/accuracy metrics |
| AdaCount | training-free zero-shot counting | similarity-guided spatial/feature adaptation | adaptation and proposal quality, not FQIE whole-instance gate | not the FQIE Safe-IIC contract | not the stated output contract | count metrics |
| FQIE-PR-IIC | frozen CountGD++ query-only rescue | frozen CountGD++ full/tile proposals + cross-view consensus | semantic YES + whole-instance YES required | non-transitive, compatibility-constrained Safe-IIC; UNKNOWN never merges | explicit UNKNOWN and partial/abstained count states | verified instances, provenance, hashes, structured evidence trace |

## Defensible novelty statement

The audit supports a method/protocol-level positioning claim, not an absolute first claim:

> FQIE studies query-only counting as evidence-grounded physical-instance verification. It combines GT-blind proposal recovery with explicit semantic and whole-instance admission, conservative identity association, and coverage-aware output states that distinguish exact, partial, and abstained counts.

This statement must remain qualified until the paper includes direct reproduced baselines and multi-dataset evaluation.

## Primary sources

- CountGD: [NeurIPS 2024 paper](https://papers.nips.cc/paper_files/paper/2024/file/57c56985d9afe89bf78a8264c91071aa-Paper-Conference.pdf)
- CountGD++: [CVPR 2026 paper supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Amini-Naieni_CountGD_Generalized_Prompting_CVPR_2026_supplemental.pdf)
- Zero-Shot Object Counting: [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_Zero-Shot_Object_Counting_CVPR_2023_paper.html)
- Point, Segment and Count: [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_Point_Segment_and_Count_A_Generalized_Framework_for_Object_Counting_CVPR_2024_paper.pdf)
- CountSE: [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_CountSE_Soft_Exemplar_Open-set_Object_Counting_ICCV_2025_paper.pdf)
- SAM3Count: [OpenReview paper](https://openreview.net/pdf/42549a61b00ae67574067e1eea5dff27637b4703.pdf)
- GroundCount: [arXiv paper](https://arxiv.org/abs/2603.10978)
- AdaCount: [arXiv paper](https://arxiv.org/abs/2607.02139)
- VLCounting: [publisher paper](https://www.sciencedirect.com/science/article/pii/S0031320326001706)

## Claim gate

The paper may claim a protocol-level distinction after adding direct citations and a reproduced comparison table. It may not claim SOTA, universal counting, certified absence, or absolute novelty from this audit alone.
