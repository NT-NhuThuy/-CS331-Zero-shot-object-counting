# FQIE / CountGD++ — Handoff tổng thể

Ngày cập nhật: 2026-11-10  
Workspace: `/mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC`

## 1. Mục tiêu nghiên cứu

Mục tiêu không phải chỉ giảm MAE bằng cách ghép nhiều model. Mục tiêu đúng là:

> Với mỗi vùng ứng viên, hệ thống xác định vùng nào thuộc tập vật thể cần đếm,
> vùng nào không thuộc target, vùng nào chưa chắc chắn; đồng thời giải thích được
> vì sao vùng được giữ hoặc bị loại.

Đây là khác biệt có ý nghĩa so với CountGD++ baseline: đầu ra mong muốn là một
tập instance có lý do quyết định, sau đó mới lấy số lượng từ tập đó.

## 2. Baseline và trạng thái chung

CountGD++ p0 là baseline mạnh và phải luôn được giữ làm đối chứng. Các nhánh sau
đều là VAL development, không phải TEST claim.

### Count Anything confirmation 300 query

File: `target_aware_counting/output/count_anything_confirm300_20261108_v1/RESULT.md`

- p0: MAE 1.1800, RMSE 2.6621, TP 185, FP 152, FN 224.
- score+witness: MAE 0.9133, RMSE 1.8092, TP 174, FP 51, FN 235.
- Giảm FP/MAE/RMSE nhưng mất 11 TP và thêm false-zero.
- Các TP mất tập trung ở cảnh nhiều instance: bench, chair, streetlight,
  necklace, traffic light.
- Witness point không tự tạo instance; chỉ giữ/xóa p0 box.

### Frozen V2 confirmation 300 query

File: `target_aware_counting/output/count_anything_v2_confirm300_20261108_v1/RESULT.md`

- p0: MAE 2.3867, RMSE 15.6993, TP 233, FP 503, FN 265.
- score-only: MAE 1.5967, RMSE 7.5501, TP 140, FP 143, FN 358.
- V2 rescue: MAE 1.5033, RMSE 7.4353, TP 231, FP 238, FN 267.
- V2 giảm FP 265, giảm mạnh MAE/RMSE, giữ false-zero và absent metrics.
- Tuy nhiên mất 2 TP, cả hai là streetlight; gate “không mất target” fail.
- Một streetlight dense bị cap rescue bỏ sót; một streetlight có 264 p0 box
  nhưng không có witness nên không biết box thật nào cần giữ.
- Quyết định: `NOT_PROMOTED`, nhưng V2 là policy an toàn nhất hiện có.

## 3. FQIE Gate 0 / H2 — ngữ nghĩa

File chính: `target_aware_counting/output/fqie_gate0_20261104_v1/GATE_STATUS.md`

### CycleGRPO

CycleGRPO là tín hiệu ngữ nghĩa mạnh nhất đã thử, nhưng chưa phải bộ lọc cuối.

File số liệu: `target_aware_counting/output/fqie_h2_lvis306_cycle_eval_20261104_v1/RESULT.md`

Trên 306 proposal / 274 query pair:

| Rule | MAE | RMSE | TP50 | False-zero | Absent mass |
|---|---:|---:|---:|---:|---:|
| No-op | 1.948 | 3.842 | 163 | 82 | 13 |
| Cycle YES | 1.701 | 3.218 | 157 | 85 | 0 |
| Alpha AND Cycle | 1.718 | 3.190 | 151 | 89 | 0 |
| Alpha OR Cycle | 1.902 | 3.797 | 163 | 83 | 0 |

Cycle giữ 158/164 strict target, loại nhầm 6 target (3.7%). MAE/RMSE và absent
mass tốt hơn, nhưng TP50 giảm 6 với bootstrap CI loại zero. Vì vậy không thể
dùng luật `Cycle=NO -> xóa box` làm bộ lọc cứng.

### Cycle identical-region query switch

File: `target_aware_counting/output/fqie_h2_lvis_paired_region_cycle_eval_20261104_v1/RESULT.md`

- Cùng một vùng thật, query đúng lớp: YES 94/100.
- Query verified-absent: YES 0/100.
- Đây là bằng chứng tốt cho class/head semantic discrimination.
- Không kiểm tra đầy đủ whole-object, part, duplicate, relation, attribute.

### Alpha / CLIP

Alpha cải thiện separation nhưng threshold chuyển giao không ổn định:

- giữ được target nhưng vẫn kích hoạt nhiều absent query;
- raw+generic margin giảm absent mass nhưng còn nhiều query active;
- không được dùng làm semantic gate độc lập.

Gate status ghi đầy đủ các con số và giới hạn trong:
`target_aware_counting/output/fqie_gate0_20261104_v1/GATE_STATUS.md`

### SigLIP semantic contrast

File: `target_aware_counting/output/fqie_semantic_contrast_pilot_20261106_v1/RESULT.md`

SigLIP target-vs-neutral có thông tin semantic, nhưng không phân biệt chắc whole
target với target part/different object. Hai chân gấu vẫn có score cao như whole
teddy bear; vùng bánh mì dưới tongs cũng bị đánh giá cao. Không dùng cutoff đơn.

## 4. Grounding / witness / multi-scale

### GroundingDINO whole-object witness

File: `target_aware_counting/output/fqie_grounding_witness_transfer_20261106_v1/RESULT.md`

- p0: MAE 0.6875, TP 29, FP 12.
- p3: MAE 3.2813, TP 29, FP 107.
- p3 + grounded witness: MAE 1.4063, TP 7, FP 3, FN 48.
- Giảm lỗi nhưng xóa quá nhiều target.
- Lỗi cơ chế: DINO box thường bao cả bowl/person thay vì carrot/cap; sai granularity.

Local crop diagnostic: `target_aware_counting/output/fqie_local_grounding_pilot_20261106_v1/RESULT.md`

Hai scale không đủ phân biệt whole-object với part/duplicate. Containment/IoU
không phải semantic instance oracle.

### Multi-scale witness

File: `target_aware_counting/output/fqie_multiscale_witness_integration_20261109_v1/RESULT.md`

- p0: TP 37, FP 351, MAE 26.6154.
- V2: TP 35, FP 165, MAE 12.1538.
- Multi-scale: TP 35, FP 174, MAE 12.8462.
- 9 box được cứu, cả 9 là false positive.
- Tile lặp lại cùng lỗi semantic; nhiều view không đồng nghĩa bằng chứng độc lập.
- Nhánh này bị loại, không tune tiếp.

## 5. Reference branch — tạm thời đóng

### Internal reference bank

File: `target_aware_counting/output/fqie_reference_bank_v2_20261109_v1/RESULT.md`

- 388 candidate native thành 245 sau NMS.
- Current top-3 phủ 17/44 GT instance.
- Frozen bank phủ 23/44.
- Tất cả 900 decoder slots phủ 31/44.
- Một số query (`streetlight`, `button`) không có candidate đúng ngay cả trong
  900 slots: selector phía sau không thể sửa candidate absence.

### SAM3 external source

File: `target_aware_counting/output/fqie_external_reference_sam3_20261109_v1/coverage.json`

SAM3 bổ sung coverage nhưng không phải semantic oracle. Source-disjoint evidence
cho thấy SAM3 hữu ích như nguồn đề xuất vùng, không được coi là nhãn đúng.

### OWLv2 coverage

File: `target_aware_counting/output/fqie_external_reference_owlv2_20261109_v1/RESULT.md`

- OWLv2 phủ 42/44 instance trên 13 query.
- CountGD++ + SAM3 trước đó phủ 29/44.
- OWLv2 bổ sung 13 instance mà hai nguồn trước bỏ sót.
- Top-20 phủ 41/44; top-40 phủ 42/44.
- Đây chỉ là proposal coverage, không phải semantic purity.

### OWLv2 top-1 intervention — bị loại

File: `target_aware_counting/output/fqie_owl_reference_intervention_20261109_v1/RESULT.md`

| Mode | MAE | RMSE | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| Text-only | 26.62 | 73.77 | 23 | 365 | 21 |
| OWLv2 top-1 reference | 59.31 | 163.00 | 25 | 788 | 19 |
| Without reference | 26.62 | 73.77 | 23 | 365 | 21 |

OWLv2 top-1 thêm 2 TP nhưng thêm 423 FP. Reference đúng về IoU vẫn có thể làm
CountGD++ kích hoạt hàng trăm distractor. Không dùng top-1 reference trực tiếp.

Code intervention đã được thêm vào:

- `CountGDPlusPlus/engine_inference.py`
- `CountGDPlusPlus/main_inference.py`
- `target_aware_counting/output/fqie_owl_reference_intervention_20261109_v1/`

Các thay đổi này hiện phục vụ diagnostic, không phải policy đã promote.

## 6. Hai lỗi cốt lõi cần giải quyết

### Lỗi A — Candidate absence

CountGD++ đôi khi không tạo ra box đúng. Mọi bộ lọc sau đó đều bất lực.

### Lỗi B — Unsafe conditioning

Một reference có thể đúng hình học nhưng làm Forward-2 sinh hàng trăm box sai.

Do đó reference selection là cần thiết nhưng chưa đủ. Cần kiểm tra tác động của
reference lên output trước khi cho phép thay đổi prediction.

## 7. Cách hiểu đúng về 6 target CycleGRPO loại nhầm

6/164 không phải lỗi nhỏ nếu claim là “biết vùng nào nên đếm”. Nó cho thấy luật
nhị phân `NO -> xóa` đang quá mạnh. Các nguyên nhân cần kiểm chứng tiếp:

- vật thể nhỏ hoặc bị che;
- cảnh đông, nhiều instance gần nhau;
- box chỉ bao một phần vật thể;
- query/category có biến thể ngôn ngữ;
- CycleGRPO không có calibration/confidence đủ tốt;
- quyết định từng box độc lập, không nhìn toàn bộ nhóm instance.

Không được tự kết luận nguyên nhân nào là đúng trước khi bóc riêng 6 case trên
ảnh/box. Nhưng hướng sửa đã rõ: `NO` không còn quyền tự động xóa trong vùng rủi
ro cao.

## 8. Solution nên phát triển tiếp

Không thêm model chỉ để thêm model. Xây một bộ chọn instance ba trạng thái:

```text
KEEP     : đưa box vào tập đếm
REMOVE   : loại vì có nhiều bằng chứng sai cùng hướng
UNKNOWN  : chưa đủ chắc, không tự ý xóa
```

Mỗi box phải có decision record:

- native CountGD++ score;
- CycleGRPO YES/NO và mức tin cậy;
- Count Anything witness;
- độ lớn box và mức chen chúc;
- overlap/containment với box khác;
- ổn định qua view/crop nếu có;
- lý do cuối cùng: giữ, loại hoặc chưa chắc.

### Quy tắc bảo thủ

- `REMOVE` chỉ khi có nhiều bằng chứng độc lập cùng hướng.
- Nếu Cycle nói NO nhưng box nhỏ, nằm trong cảnh đông, hoặc native score không
  thấp: chuyển UNKNOWN/giữ lại.
- Nếu một quyết định làm giảm quá nhiều box trong một query dense: kích hoạt
  density safeguard, không xóa hàng loạt.
- Reference ngoài chỉ dùng tạo candidate/định vị; không cho phép nó tự kích hoạt
  Forward-2 nếu chưa qua stress test.
- Không chắc thì fallback về V2 hoặc p0, tùy policy đã khóa.

### Claim mục tiêu

Claim nên hướng tới:

> Hệ thống xây dựng một tập instance được chọn có điều kiện và giải thích được;
> số đếm là tổng số instance trong tập đó, thay vì chỉ là số box sau một ngưỡng.

Để claim được, cần báo cáo cả:

- count MAE/RMSE;
- TP/FP/FN;
- false-zero;
- absent mass;
- recall target;
- tỷ lệ quyết định KEEP/REMOVE/UNKNOWN;
- lý do quyết định trên từng box;
- hiệu quả trên cohort source-disjoint đã khóa trước nhãn.

## 9. Thứ tự đọc nhanh ở conversation mới

1. `target_aware_counting/output/fqie_gate0_20261104_v1/GATE_STATUS.md`
2. `target_aware_counting/output/count_anything_v2_confirm300_20261108_v1/RESULT.md`
3. `target_aware_counting/output/fqie_h2_lvis306_cycle_eval_20261104_v1/RESULT.md`
4. `target_aware_counting/output/fqie_h2_lvis_paired_region_cycle_eval_20261104_v1/RESULT.md`
5. `target_aware_counting/output/count_anything_confirm300_20261108_v1/RESULT.md`
6. `target_aware_counting/output/fqie_grounding_witness_transfer_20261106_v1/RESULT.md`
7. `target_aware_counting/output/fqie_local_grounding_pilot_20261106_v1/RESULT.md`
8. `target_aware_counting/output/fqie_multiscale_witness_integration_20261109_v1/RESULT.md`
9. `target_aware_counting/output/fqie_reference_bank_v2_20261109_v1/RESULT.md`
10. `target_aware_counting/output/fqie_external_reference_owlv2_20261109_v1/RESULT.md`
11. `target_aware_counting/output/fqie_owl_reference_intervention_20261109_v1/RESULT.md`
12. `FQIE_V2_RESEARCH_IMPLEMENTATION_20261103.md`

## 10. Trạng thái quyết định

- Baseline CountGD++ p0: giữ làm đối chứng.
- V2 Count Anything rescue: tốt về MAE/FP nhưng `NOT_PROMOTED` vì mất TP.
- CycleGRPO: semantic signal mạnh, chưa dùng hard gate.
- Alpha/SigLIP: feature phụ, chưa làm gate độc lập.
- GroundingDINO witness: dừng vì sai granularity và mất TP.
- Multi-scale witness: rejected; không phải bằng chứng độc lập.
- OWLv2: tốt cho candidate coverage, không được đưa trực tiếp làm reference.
- Reference direct intervention: đóng tạm thời.
- Nhánh đang cần phát triển: decision mechanism KEEP/REMOVE/UNKNOWN có giải thích,
  bảo vệ target trong cảnh đông và có fallback bảo thủ.

Không được gọi bất kỳ kết quả development nào là final benchmark/test claim khi
chưa có cohort source-disjoint mới và policy khóa trước khi xem nhãn.
