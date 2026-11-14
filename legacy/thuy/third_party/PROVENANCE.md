# Provenance and licensing

This table is a release gate. The repository license applies only to original ZOC-authored files that the project is authorized to license. It does not relicense upstream code.

| Component | Required verification | Current status |
|---|---|---|
| CountGD++ | Upstream URL, exact commit, upstream license, local diff | pending verification; source is archived for research only |
| GroundingDINO | Upstream URL, commit, license, local modifications | pending verification |
| SAM2/SAM3 | Upstream URL, checkpoint/model license, commit, local modifications | not vendored; runtime artifacts excluded |
| Alpha-CLIP / CycleGRPO | Upstream URL, license, commit and dependency terms | not vendored; runtime artifacts excluded |
| FQIE / reference correspondence | Originating ZOC documents, author ownership, dependencies | experimental ZOC modules; release ownership review required |
| ZOC-authored modules | Author/owner confirmation and chosen license | Apache-2.0 applies only after owner confirmation |

Until this table is completed, do not claim that the whole repository is Apache-2.0 licensed.
