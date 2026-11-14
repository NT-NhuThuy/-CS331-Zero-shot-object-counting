# FQIE / CountGD++ — Tình hình hiện tại và roadmap xây dựng claim paper

> **Cập nhật 2026-11-16:** Safe-IIC rescue-only đã được thay bằng bidirectional evidence controller đã freeze. V13 development và V14 source-disjoint locked holdout đều có bootstrap CI loại 0. Trên V14, MAE giảm 1.7611 → 1.1000, absent MAE 0.1667 → 0.0000, 22 thắng/3 thua, sign p=0.0001565. Không tune theo V14; việc tiếp theo là external-dataset validation, baseline comparisons, efficiency và manuscript.

Ngày cập nhật: 2026-11-12  
Workspace: `/mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC`

## 1. Mục tiêu bài toán

Mục tiêu là xây dựng một inference cascade cho CountGD++ frozen, nhằm xử lý các query mà full-image CountGD++:

- trả về zero dù target có mặt;
- hoặc có bằng chứng yếu/thiếu proposal;
- đặc biệt trong setting query-only, không đọc GT trong lúc inference.

Ràng buộc phương pháp:

- không fine-tune CountGD++;
- không dùng GT để chọn prompt, threshold, tile, merge hoặc decision;
- mọi artifact inference phải được hash-lock trước khi đánh giá hậu nghiệm;
- UNKNOWN không được tự động xem là NO, không promote và không merge.

## 2. Các nhánh đã thử

### 2.1. V3 — Presence certificate

Đây là nhánh xác nhận target có hiện diện bằng evidence độc lập.

- 24 PRESENT;
- 156 UNKNOWN;
- không có absence certificate mạnh;
- phù hợp làm presence evidence hẹp, không phải counting method.

Kết luận: không dùng làm method chính.

### 2.2. V4 — Conservative instance selector

Multi-reference/visual selector lọc proposal CountGD++.

- full baseline MAE: `3.128`;
- selector MAE: `2.933`;
- present MAE: `5.833 → 5.511`;
- absent MAE: `0.422 → 0.356`;
- 9 query cải thiện, 1 query xấu.

Có tín hiệu tốt nhưng chưa đủ để làm core method, và reference gate chưa ổn định khi ghép với rescue branch.

### 2.3. V5 — Raw tiled proposal union

Dùng full image + nhiều tile rồi union raw detections.

- coverage tăng;
- nhưng MAE tăng mạnh `2.417 → 4.900`;
- present MAE `4.8 → 9.489`;
- absent MAE `0.033 → 0.311`.

Failure mode: crop-specific pseudo-exemplar amplification và duplicate/part detections.

Kết luận: raw tile union bị loại.

### 2.4. V6 — Zero-only two-scale tile rescue

Chỉ rescue query full-image count bằng zero, dùng tile 60%/75% và Cycle YES/NO.

- full MAE: `1.817`;
- V6 MAE: `1.844`;
- 7/10 rescued query cải thiện;
- failures: tablecloth, radiator, knee pad do part/texture/multiple-object crop.

Kết luận: cần whole-instance và identity control.

### 2.5. V7/V8 — Proposal Recovery + Instance Identity (method hiện tại)

Pipeline:

```text
frozen CountGD++ full-image
→ zero-only multi-view tile proposals
→ cross-view IoU consensus
→ semantic YES/NO
→ whole-instance YES/NO
→ pairwise same-physical-instance
→ conservative union-find
→ final count
```

`UNKNOWN` không được promote/merge.

#### V7

- cohort: 180 query units, 90 present + 90 verified absent;
- 1,072 tile units;
- 215 consensus candidates;
- 86 semantic + whole-instance eligible candidates;
- 1,063 pair checks;
- 660 SAME=YES edges;
- 22 promoted physical instances trên 13 query;
- all MAE: `2.044 → 1.967`;
- present MAE: `4.000 → 3.844`;
- absent MAE: `0.089 → 0.089`;
- 11 query cải thiện, 0 query xấu.

#### V8

- cohort mới, source-disjoint với các nhánh trước;
- 1,008 tile units;
- 223 consensus candidates;
- 123 semantic + whole-instance eligible candidates;
- 2,686 pair checks;
- 1,482 SAME=YES edges;
- 17 promoted physical instances trên 12 query;
- all MAE: `3.078 → 2.994`;
- present MAE: `5.833 → 5.667`;
- absent MAE: `0.322 → 0.322`;
- 11 query cải thiện, 0 query xấu.

V7/V8 là bằng chứng hiện tại tốt nhất cho method chính.

### 2.6. Reference consistency ablation

Reference branch được chạy thật trên V8:

- 223 reference crops;
- 78/223 candidate qua target-vs-hard-confuser gate trên 3 views;
- identity components đầu vào: 17;
- components còn lại sau reference gate: 11.

Ablation V8:

| Variant | All MAE | Present MAE | Absent MAE | Improved / Worsened |
|---|---:|---:|---:|---:|
| A — baseline | 3.078 | 5.833 | 0.322 | 0 / 0 |
| B — raw proposal-only | 3.128 | 5.767 | 0.489 | 8 / 5 |
| C — proposal + identity | **2.994** | 5.667 | **0.322** | **11 / 0** |
| D — reference-only | 2.778 | **5.111** | 0.444 | 9 / 1 |
| E — identity + reference gate | 3.017 | 5.711 | 0.322 | 9 / 0 |

Interpretation:

- B chứng minh raw proposal union gây overcount;
- C là variant an toàn nhất;
- D giảm MAE nhưng làm absent error tăng;
- E kém C vì reference gate loại mất một số identity components đúng.

Quyết định: strict reference gate không đưa vào primary method; chỉ báo cáo như ablation/negative result.

#### Reference soft-veto — nhánh phát triển mới

Strict gate đã được sửa thành soft-veto:

- margin âm ở một view: UNKNOWN, giữ candidate;
- chỉ veto khi cả 3 view của toàn bộ identity component đều âm;
- reference không tự sinh count;
- reference chỉ veto component đã được IIC freeze.

Diagnostic V8:

- strict gate giữ 11/17 identity components;
- soft-veto giữ 12/17;
- strict E MAE: `3.017`;
- soft-veto E MAE: `3.011`;
- primary C MAE: `2.994`;
- absent MAE soft-veto vẫn `0.322`, không tăng;
- soft-veto vẫn chưa vượt C trên V8.

Soft-veto sẽ được đánh giá trên cohort V10 fresh trước khi quyết định có đưa vào optional enhancement hay không.

### 2.7. V9 — Dense ownership branch

V9 thử 3×3 tile 50% overlap và deterministic tile ownership, không dùng pairwise identity.

- 1,179 dense tiles;
- 218 ownership detections;
- baseline MAE: `2.528`;
- V9 MAE: `3.061`;
- present MAE: `5.056 → 5.778`;
- absent MAE: `0.000 → 0.344`;
- improved: 7 query;
- worsened: 15 query.

Failure mode: ownership loại duplicate nhưng không loại false tile detections. V9 bị loại khỏi method.

Diagnostic thêm cho thấy V8 pairwise identity có over-merge trong dense scenes:

- 1,482 SAME=YES edges;
- median box IoU = 0;
- mean IoU ≈ 0.013;
- q153: 49 eligible candidates bị merge thành 1;
- q179: 55 eligible candidates bị merge thành 3.

Đây là giới hạn dense counting cần báo cáo trung thực, không được che bằng threshold tuning hậu nghiệm.

## 3. Method được chốt hiện tại

### Primary method: FQIE-PR-IIC

Tên làm việc:

**FQIE: Frozen Query-only Evidence with Proposal Recovery and Instance Identity Control**

Core method:

```text
CountGD++ frozen full-image inference
→ zero/low-evidence tile proposal recovery
→ cross-view consensus
→ semantic target check
→ whole-instance check
→ conservative physical-identity control
→ final count
```

Reference strict gate không nằm trong primary cascade. Soft-veto là optional branch đang validation. Dense raw ownership không nằm trong primary cascade.

## 4. Claim được phép viết

Claim chính hiện tại:

> A GT-blind cross-view proposal-recovery and whole-instance identity cascade improves zero/low-evidence CountGD++ counting on two source-disjoint LVIS audits without worsening absent-query error.

Diễn giải tiếng Việt:

> Một cascade GT-blind gồm phục hồi proposal đa view, kiểm tra whole-instance và kiểm soát identity vật thể giúp cải thiện CountGD++ trên các query zero/low-evidence ở hai cohort LVIS source-disjoint mà không làm tăng lỗi trên query absent.

Claim phụ có thể viết:

- raw tiled union không an toàn vì gây overcount;
- reference-only gate có thể giảm MAE nhưng làm absent error tăng;
- whole-instance + identity control là thành phần quyết định để giữ safety;
- method hiện tại phù hợp conservative rescue, không phải complete dense counting.

## 5. Claim chưa được phép viết

Chưa đủ bằng chứng để claim:

- state-of-the-art counting;
- universal/general counting trên mọi dataset;
- complete recovery cho dense scenes;
- certified absence;
- reference consistency luôn cải thiện method;
- method không bao giờ overcount;
- superiority trước tất cả baselines.

## 6. Roadmap từ hiện tại tới paper

### Phase 1 — Freeze code (đang hoàn tất)

- freeze V7/V8 PR+IIC code;
- freeze prompt, tile scales, IoU consensus, UNKNOWN policy;
- freeze hash của model/input/artifact;
- đánh dấu V9 và reference gate là negative/ablation branches.

Deliverable: `FQIE-PR-IIC release candidate`.

### Phase 2 — Independent multi-dataset evaluation

Chạy nguyên protocol, không tuning trên test:

1. LVIS source-disjoint cohort thứ ba;
2. FSC147;
3. OmniCount/Fruits hoặc dataset counting có annotation;
4. Blood-cell hoặc một dataset domain-shift khác.

Mỗi dataset cần báo:

- baseline CountGD++;
- FQIE-PR-IIC;
- MAE, RMSE, bias;
- exact-count accuracy;
- present recall;
- absent false-positive rate;
- số query được rescue;
- runtime và peak VRAM.

### Phase 3 — Statistical analysis

- paired per-query bootstrap 95% CI cho MAE delta;
- sign test hoặc Wilcoxon paired test;
- báo riêng present/absent;
- không gộp absent và present thành một con số duy nhất để che trade-off.

### Phase 4 — Required ablations

Trên một validation split riêng:

- baseline;
- raw proposal-only;
- proposal + semantic;
- proposal + whole-instance;
- proposal + identity;
- proposal + reference;
- full cascade.

V7/V8 chỉ là held-out audits; không dùng để chọn threshold mới.

### Phase 5 — Failure analysis

Bắt buộc có các nhóm:

- zero nhưng dense count cao;
- part/texture false positives;
- multiple-object crop;
- pairwise identity over-merge;
- reference gate loại nhầm evidence;
- absent-query false positives.

Mỗi nhóm phải có hình minh họa, số lượng, nguyên nhân và giới hạn claim.

### Phase 6 — Paper structure

1. Introduction: zero/low-evidence failure của query-only counting.
2. Related work: open-vocabulary counting, tile rescue, visual verification, instance identity.
3. Method: PR, consensus, whole-instance, identity graph, abstention policy.
4. Protocol: frozen checkpoint, GT-blind inference, hash locks.
5. Results: multi-dataset table.
6. Ablation: A–E và negative branches.
7. Failure analysis: dense/over-merge/absent.
8. Limitations: not complete dense counting, no absence certificate.
9. Conclusion: conservative evidence-aware rescue.

## 7. Điều kiện để chốt paper

Chỉ chốt primary claim khi thỏa cả bốn điều kiện:

1. FQIE-PR-IIC cải thiện trên ít nhất ba cohort/dataset độc lập;
2. absent error không tăng đáng kể;
3. confidence interval của MAE delta không mâu thuẫn với claim;
4. mọi ablation và negative branch được báo cáo minh bạch.

Nếu chỉ V7/V8 vẫn tốt còn dataset khác không ổn, claim phải thu hẹp thành:

> conservative zero-evidence rescue on LVIS-style query-only counting.

## 8. Việc triển khai ngay tiếp theo

1. Giữ FQIE-PR-IIC là primary release candidate.
2. Chạy soft-veto trên cohort V10 fresh; không tune trên V8.
3. Không dùng raw dense ownership V9 trong method chính.
4. Tạo evaluation runner chung cho FSC147/OmniCount/Blood-cell.
5. Chạy dataset thứ ba trước khi viết claim SOTA.
6. Sau khi có bảng multi-dataset, hoàn thiện paper draft và figures.
