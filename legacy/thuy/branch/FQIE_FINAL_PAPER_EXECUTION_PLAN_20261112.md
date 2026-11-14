# FQIE — Final execution plan to paper

> **Execution update 2026-11-16:** bidirectional development, locked source-disjoint holdout, paired uncertainty, failure audit và rescue/rejection/full ablations đã hoàn tất. Gate kế tiếp là external-dataset validation với method giữ nguyên; thêm một version cùng distribution không thay thế được gate này.

External execution details, baselines, scalability routing, metrics, and claim gates are frozen in `FQIE_A_PAPER_EXTERNAL_EVALUATION_PLAN_20261116.md`.

Ngày chốt kế hoạch: 2026-11-12

## 0. Mục tiêu cuối

Xây dựng và đánh giá một inference cascade GT-blind cho frozen CountGD++ có khả năng:

1. phục hồi target khi full-image inference trả zero/low evidence;
2. kiểm tra candidate là đúng target và whole physical instance;
3. không đếm duplicate/part/background khi evidence chưa đủ;
4. trả về verified instance set, không chỉ một scalar count;
5. abstain khi không thể xác nhận.

Primary method tạm chốt:

```text
CountGD++ frozen
→ zero/low-evidence proposal recovery
→ cross-view consensus
→ semantic target verification
→ whole-instance verification
→ Safe-IIC identity control
→ verified instance set
→ count + evidence trace
```

Reference consistency và raw dense ownership không nằm trong primary cascade hiện tại.

## 1. Những gì đã freeze

### Evidence đã có

- V7 source-disjoint audit: all MAE `2.044 → 1.967`, present MAE `4.000 → 3.844`, absent không đổi, 11 improved/0 worsened.
- V8 source-disjoint audit: all MAE `3.078 → 2.994`, present MAE `5.833 → 5.667`, absent không đổi, 11 improved/0 worsened.
- Raw proposal-only làm absent error tăng.
- Strict reference gate làm full cascade kém PR+IIC.
- Soft-veto an toàn hơn strict gate nhưng chưa vượt PR+IIC trên V8.
- Dense ownership V9 bị loại: MAE `2.528 → 3.061`, absent error tăng.

### Không được thay đổi khi báo cáo V7/V8

- frozen CountGD++ checkpoint;
- tile/proposal protocol;
- consensus rule;
- prompt và UNKNOWN policy;
- identity policy;
- cohort membership;
- hash locks.

Mọi thay đổi sau đây phải mang version mới, không được gọi là V7/V8 continuation.

## 2. Phase A — Literature and positioning audit

### Mục tiêu

Xác định novelty ở method/protocol level, không claim novelty của tile, NMS, semantic prompt hoặc identity matching riêng lẻ.

### Việc phải làm

- đọc primary papers của CountGD++, SAM3Count, ValidCounter, AdaCount, CountSE, LAPE, GroundCount, CounterCount và các công trình gần nhất;
- lập bảng mapping: input, trigger, proposal source, semantic check, identity policy, abstention, output evidence;
- xác định chính xác FQIE khác ở đâu:
  - zero/low-evidence trigger;
  - frozen open-world query-only setting;
  - whole-instance evidence gate;
  - abstaining verified-instance set.

### Gate A

Không được dùng câu “first/novel/unprecedented” nếu chưa có bảng đối chiếu và citation trực tiếp.

## 3. Phase B — Freeze task specification

### Scope v1

Chỉ hỗ trợ:

- category query;
- whole physical instance;
- absent target;
- semantic confuser;
- part/texture/background rejection;
- UNKNOWN/abstention.

Chưa mở rộng ngay sang màu sắc, quan hệ không gian và compositional multi-attribute nếu chưa có annotation.

### Output schema

Mỗi query phải lưu:

```json
{
  "count": 0,
  "accepted_instances": [],
  "rejected_regions": [],
  "unknown_regions": [],
  "evidence_trace": [],
  "abstained": false
}
```

Mỗi accepted instance cần candidate box/mask, semantic status, whole-instance status, identity/component ID và provenance hash.

### Gate B

Schema và ambiguity policy phải freeze trước khi tạo hard-case annotations.

## 4. Phase C — Build hard-case evidence benchmark

### Dataset structure

Tạo validation/test subsets với các nhóm:

1. zero-evidence target;
2. low-evidence target;
3. semantic confuser;
4. attribute/category mismatch nếu dataset hỗ trợ;
5. part-whole confusion;
6. multiple-object crop;
7. nearby separate instances;
8. duplicate multi-view proposal;
9. absent target;
10. dense scene.

### Annotation

Mỗi case cần:

- GT count;
- GT box/mask;
- target instance ID;
- hard-negative region/type;
- part/background label;
- ambiguous/unknown label;
- source split and provenance.

### Gate C

Không dùng test GT để chọn prompt, threshold, crop, merge hoặc trigger. Validation và test phải tách nguồn.

## 5. Phase D — Safe-IIC implementation

### Vấn đề phải sửa

V8 cho thấy pairwise VLM SAME có median IoU bằng 0 và có thể over-merge hàng chục candidate thành một component.

### Thiết kế Safe-IIC

- pairwise edge không được tự động tạo transitive merge;
- candidate cần semantic YES + whole YES trước khi vào graph;
- edge cần local compatibility và geometry consistency;
- edge thiếu evidence là UNKNOWN;
- component lớn phải có giới hạn/cảnh báo;
- không merge nếu một cạnh đơn lẻ là cầu nối duy nhất;
- báo false merge và false split riêng.

### Gate D

Safe-IIC phải được đánh giá trên validation hard cases trước khi chạy test. Không chọn rule từ V7/V8 test GT.

## 6. Phase E — Complete ablation

Trên validation/test tách biệt, chạy:

| Variant | Thành phần |
|---|---|
| A | frozen CountGD++ |
| B | raw proposal recovery |
| C | semantic verification |
| D | whole-instance verification |
| E | semantic + whole-instance |
| F | E + Safe-IIC |
| G | full evidence trace + abstention |
| H | reference soft-veto optional |

Metrics:

- MAE, RMSE, bias;
- exact-count accuracy;
- target precision/recall;
- hard-negative rejection;
- box/mask IoU;
- false merge, false split;
- absent false-positive;
- UNKNOWN rate;
- risk–coverage;
- runtime, calls, peak VRAM.

### Gate E

Primary method chỉ được chọn nếu F/G cải thiện C/E mà không làm absent safety hoặc false merge xấu đi đáng kể.

## 7. Phase F — Multi-dataset evaluation

Chạy cùng frozen protocol trên:

1. LVIS source-disjoint cohort thứ ba;
2. FSC147;
3. OmniCount/Fruits;
4. Blood-cell hoặc domain-shift dataset.

Nếu chỉ LVIS cải thiện, claim phải giới hạn là LVIS-style query-only rescue. Không dùng chữ general-purpose.

## 8. Phase G — Statistical validation

- paired per-query bootstrap 95% CI;
- Wilcoxon/sign test cho MAE delta;
- present/absent tách riêng;
- zero-baseline subset tách riêng;
- risk–coverage curve;
- runtime/accuracy trade-off;
- report all negative branches.

### Gate G

Claim improvement chỉ được dùng nếu confidence interval và paired analysis không mâu thuẫn với hướng kết luận.

## 9. Phase H — Paper writing

### Paper structure

1. Introduction: zero/low-evidence failure trong query-only counting.
2. Related Work: open-world counting, adaptive crop, semantic verification, identity association, abstention.
3. Problem formulation: verified physical-instance counting.
4. Method: proposal recovery, consensus, whole-instance, Safe-IIC, evidence trace.
5. Protocol: frozen checkpoint, GT-blind inference, provenance/hash lock.
6. Results: multi-dataset table.
7. Ablation: A–H.
8. Failure analysis: dense, confuser, part-whole, over-merge, absent.
9. Limitations: not complete dense counting, not certified absence.
10. Conclusion.

### Figure/table bắt buộc

- pipeline diagram;
- accepted/rejected/unknown evidence visualization;
- baseline vs FQIE per-query paired plot;
- ablation table;
- risk–coverage plot;
- failure gallery;
- runtime/VRAM table;
- novelty comparison table với related works.

## 10. Claim ladder

### Claim tối thiểu hiện đã có thể bảo vệ

> A GT-blind cross-view proposal-recovery and whole-instance identity cascade improves zero/low-evidence CountGD++ counting on two source-disjoint LVIS audits without worsening absent-query error.

### Claim mục tiêu sau khi hoàn thành Phase F/G

> FQIE is an evidence-grounded, abstention-aware query-only counting cascade that recovers missed target instances from a frozen open-world counter and returns a verifiable physical-instance set across multiple datasets.

### Không được claim nếu chưa có evidence

- SOTA;
- universal counting;
- complete dense recovery;
- absolute novelty;
- certified absence;
- no-overcount guarantee.

## 11. Thứ tự thực thi cuối cùng

1. Hoàn tất literature audit.
2. Freeze task schema và output evidence schema.
3. Xây hard-case evidence benchmark.
4. Implement/test Safe-IIC trên validation split.
5. Chạy ablation A–H.
6. Chạy LVIS cohort thứ ba và 2–3 dataset domain khác.
7. Tính CI/paired tests/runtime.
8. Chọn claim theo evidence thực tế.
9. Viết paper, figures, tables và limitations.
10. Chỉ sau đó mới quyết định venue/submission.

## 12. Quyết định điều hành

- Primary release candidate: PR+IIC, sau khi Safe-IIC được xác thực.
- Reference: giữ như optional/ablation, soft-veto đang validation.
- V9 dense ownership: negative branch, không dùng trong primary.
- Không mở thêm branch mới nếu chưa pass Gate C–E.

## 13. P0 amendments before Safe-IIC coding

Các điều kiện sau phải được khóa trước khi giao coder triển khai Safe-IIC hoặc benchmark test:

### 13.1. Literature audit additions

Bổ sung LAPE và VLCounting vào bảng đối chiếu, cùng CountGD++, SAM3Count, ValidCounter, CountSE, AdaCount, GroundCount và CounterCount.

Mỗi công trình phải có các cột:

- zero-output recovery;
- whole physical-instance verification;
- verified instance set;
- UNKNOWN/abstention;
- evidence trace;
- same-backbone hay khác-backbone.

Không claim novelty của semantic filtering, adaptive tiling, grounding hoặc identity matching riêng lẻ.

### 13.2. Benchmark compatibility

- LVIS: benchmark chính cho boxes/masks, semantic confuser, whole-instance và identity.
- FSC147: counting và center localization; không báo cáo mask IoU/full-box correctness nếu không bổ sung annotation.
- OmniCount-191: multi-category/query evaluation theo annotation thật; không đồng nhất với OmniCount-Fruit.
- Blood-cell/domain shift: chỉ dùng sau feasibility audit về model compatibility và annotation quality.

Hard-case selection phải được freeze từ baseline trước khi chạy FQIE test. Zero/low-evidence, confuser, part-whole, nearby và absent phải có định nghĩa operational trước.

### 13.3. Safe-IIC implementation contract

- Admission: semantic YES + whole-instance YES.
- Coordinates: original-image coordinates, provenance của view/tile bắt buộc.
- Pairwise evidence không được tự động tạo transitive Union-Find.
- Cluster phải thỏa compatibility toàn cục; bridge edge đơn lẻ không đủ.
- Geometry chỉ là feasibility constraint, không tự xem là identity proof.
- UNKNOWN không merge.
- Báo false merge và false split riêng.
- Component lớn phải có diagnostic và có thể abstain.

### 13.4. Output/evidence contract

Mỗi query phải phân biệt exact và partial:

```json
{
  "count": 3,
  "count_lower_bound": 3,
  "count_upper_bound": 4,
  "count_status": "partial",
  "accepted_instances": [],
  "rejected_regions": [],
  "unknown_regions": [],
  "evidence_trace": [],
  "query_abstained": false
}
```

`evidence_trace` là structured provenance record, không phải free-form VLM explanation. Evidence trace được ghi cho mọi ablation; nó không phải một module tạo improvement.

### 13.5. Primary endpoints and statistics

Primary outcomes:

1. instance-level recovery/correctness trên zero-present queries;
2. false promotion trên zero-absent queries.

Secondary outcomes: MAE, RMSE, bias, exact count, localization, false merge/split, UNKNOWN/risk–coverage, runtime/VRAM.

Bootstrap theo image ID hoặc source-image cluster, không mặc định mỗi query unit độc lập. Safety phải dùng non-inferiority margin được định trước; “không thấy khác biệt có ý nghĩa” không đủ để claim safety equivalence.

### 13.6. Baseline contract

Ghi rõ checkpoint, prompt, pseudo-exemplar configuration, inference parameters, model version, preprocessing và hash. `num_pseudo_exemplars=0` phải được mô tả là operating point cố định hay baseline recommendation; nếu thay đổi phải tạo experiment version mới.

## 14. Contribution ladder

### Level 1 — đã có bằng chứng

Frozen CountGD++ + GT-blind cross-view recovery + whole-instance/identity control cải thiện hai LVIS audits source-disjoint mà không tăng absent MAE trên các cohort đó.

### Level 2 — cần Safe-IIC validation

Evidence-aware instance association giảm false merge/false split ở instance-level.

### Level 3 — mục tiêu paper

FQIE trả về verifiable, abstention-aware physical-instance set trên nhiều dataset tương thích annotation.

Level 3 không thể được chứng minh chỉ bằng JSON schema; cần region/identity ground truth và metrics tương ứng.

## 15. Official phase transition

### Exploration phase: DONE

Các nhánh chính đã được thử và phân loại:

- PR+IIC: có tín hiệu tích cực trên V7/V8;
- raw tile union: negative;
- dense ownership: negative;
- strict reference gate: không phù hợp primary;
- reference soft-veto: an toàn hơn nhưng chưa vượt PR+IIC;
- pairwise identity: phát hiện over-merge trong dense scenes.

Không mở thêm branch method mới để tiếp tục chase MAE.

### Validation phase: STARTED

Từ thời điểm này, mọi thay đổi phải trả lời một giả thuyết đã đăng ký trước và phải có validation split riêng. Không được dùng V7/V8/V9 để chọn lại threshold hoặc hợp thức hóa claim mới.

## 16. Vì sao MAE cải thiện chưa lớn

Mức cải thiện hiện tại khoảng 2.7–3.9% trên MAE tổng không phải bằng chứng rằng hướng nghiên cứu thất bại:

- CountGD++ đã mạnh trên query non-zero;
- FQIE chỉ tác động lên zero/low-evidence subset;
- absent queries không có room để giảm count error, chỉ đo safety;
- rescue hiện mới khôi phục một phần instance, chưa giải quyết dense multiplicity;
- identity over-merge chưa được chứng minh ở instance-level.

Vì vậy MAE tổng chỉ là Level 1 evidence. Contribution paper phải được chứng minh thêm bằng semantic correctness, localization, identity correctness, recovery và abstention safety.

## 17. Validation gates bắt buộc

Không được viết claim Level 2 nếu chưa có:

- Safe-IIC contract và implementation test;
- hard-case annotations cho target, hard negative, part/whole và instance ID;
- false merge/false split metrics;
- accepted/rejected/unknown audit;
- paired bootstrap theo image/source cluster;
- non-inferiority test cho absent false promotion.

Không được viết claim Level 3 nếu chưa có:

- ít nhất một benchmark instance-level chính;
- ít nhất hai dataset transfer với metric phù hợp annotation;
- runtime/VRAM/call-cost report;
- external baseline comparison cùng protocol hoặc giải thích rõ khác protocol.

## 18. Research objective đã khóa

Mục tiêu không còn là tối đa hóa một scalar MAE. Mục tiêu là:

> Every counted object must have auditable evidence that it matches the requested category, represents one whole physical instance, and is not counted twice; when evidence is insufficient, the system must abstain.

Do đó, mỗi instance được đếm phải có:

- category evidence;
- whole-instance evidence;
- image-region provenance;
- identity/component evidence;
- accepted/rejected/unknown status.

Đây là nguyên tắc xuyên suốt từ validation tới paper.
