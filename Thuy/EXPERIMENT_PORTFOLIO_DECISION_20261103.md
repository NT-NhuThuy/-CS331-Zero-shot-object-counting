# Báo cáo tổng kết các hướng CountGD++ đã thử và quyết định nghiên cứu tiếp

**Ngày:** 03/11/2026. **Phạm vi:** các artifact A1–A14 và B0 đã đọc trong `target_aware_counting/output/`; tất cả số liệu dưới đây là development/validation, **không phải TEST**. **Mục tiêu không đổi:** ảnh + text query bất kỳ → đúng tập instance vật lý thỏa query, count bằng số instance; query verified-absent phải trả 0. Không fine-tune hoặc huấn luyện thêm theo quyết định hiện tại của nhóm.

## Tóm tắt quyết định

**Không bỏ toàn bộ quá khứ.** A4 là thành phần cứu false-zero đã lặp lại trên FSC VAL; B0 có TP bổ sung trên cùng cohort A8; A6 chứng minh còn dư địa ở M0-positive, dù chưa đủ bền; A9 cho biết crop nhỏ hơn vẫn có ích trong ít ca. Các thất bại A2/A5/A7/A10/A11/A14 xác định chính xác ba nút thắt: chọn đúng target, không đếm trùng và không tạo count mass khi query absent. Chúng là đối chứng và bằng chứng về cơ chế, không phải sản phẩm final.

**Chưa có solution final.** Chưa có phương pháp nào đồng thời có gain count/localization ổn định ở nhiều category và kiểm soát verified-absent. Không cần bắt đầu lại từ số 0: bước kế tiếp là phân tích paired A–B trên cache hợp lệ, kiểm tra nguồn candidate và lỗi semantic/duplicate trước khi khóa một luật hybrid mới trên dữ liệu khác.

## 1. Quy tắc đọc số liệu

- Chỉ so *trực tiếp* các cột trong cùng hàng cohort. Không lấy MAE ở A4 pooled300 so với B0 primary100 để xếp hạng.
- `point TP/FP` là matching một-một theo điểm nằm trong box. Nó **không** xác minh box extent, mask hay semantic identity đầy đủ.
- `source-disjoint` giảm trùng ảnh giữa một số cohort FSC VAL, nhưng mọi FSC VAL đã xem vẫn là development. Bootstrap CI theo source chỉ mô tả độ bất định trong cohort tương ứng.
- `verified-absent` ở LVIS dùng nhãn âm tường minh. FSC category phụ không annotation không mặc nhiên là absent. Tập FSC absent111 cũ có M0 dương cả 111 cặp, vì vậy không kiểm tra an toàn của trigger `M0=0`.
- Số `count mass` trên absent là tổng box sai được trả; activation rate đơn độc có thể che một outlier hàng trăm box.

## 2. Nhánh A: cơ chế và kết quả

| Bước | Câu hỏi / cohort | Kết quả chính | Quyết định |
|---|---|---|---|
| **A1** positional reseeding | FSC primary100 cũ + verified-absent111 cũ | Primary MAE M0 **19.83** → R_TILE_40 **19.84**; TP **4077 → 4067**, FP **478 → 495**, false-zero **14 → 14**. Absent mass **6942 → 6944**, positive **111/111** giữ nguyên. | Không có incremental value. Đổi reference vị trí không tạo coverage hữu ích. [Báo cáo](target_aware_counting/output/spatial_reseeding_fsc111_20261102_v1/RESULT.md). |
| **A2** lấy toàn bộ native reference bank khi zero | Cùng primary100 cũ; exploratory, hai trigger | MAE B0 **18.47 → 13.96**, RMSE **93.19 → 49.05**, nhưng FP **505 → 895**. Một ca `2303` cứu 680/885; một ca `11771` trả **495** khi GT **133**. Bỏ ca `2303` thì A2 tệ hơn B0. | Bác bỏ all-reference fallback; outlier và overcount. [Báo cáo](target_aware_counting/output/zero_count_native_bank_fsc111_20261102_v1/RESULT.md). |
| **A3** score gate chỉ ở full image | 100 FSC VAL source khác | M0 và hai biến thể cùng MAE/RMSE **18.32/65.72**; **0 trigger**, false-zero **9 → 9**. | Full-image score không phát hiện tín hiệu tile. [Báo cáo](target_aware_counting/output/native_bank_score_holdout_20261102_v1/RESULT.md). |
| **A4** tile-native readout khi M0=0 | 100 + 200 FSC VAL source-disjoint | Pooled300: MAE **17.07 → 14.95**, RMSE **50.28 → 47.37**, TP **12869 → 13489**, FP **1814 → 1835**, false-zero **32 → 7**. Paired ΔMAE **−2.117**, CI95 **[−3.277,−1.130]**; ΔRMSE **−2.917**, CI **[−5.789,−1.084]**. | Thành phần positive zero-count recovery tốt nhất và rõ cơ chế; chưa xử lý M0-positive hay absence. [Xác nhận](target_aware_counting/output/tile_readout_confirmation_20261102_v1/RESULT.md). |
| **A5** thêm mọi box từ bốn tile trên mọi query | 100 FSC VAL mới | A4 MAE **15.29** → A5 **22.24**; A5 thêm **2241 box/95 query**, tăng TP **+460** nhưng FP **+1321** so A4. | Bác bỏ union add-only không có instance gate. [Báo cáo](target_aware_counting/output/a5_full_tile_union_val_20261102_v1/RESULT.md). |
| **A6** consensus tile cho M0-positive | Pilot100 + confirm200 FSC VAL | Pooled300: A4 MAE/RMSE **18.17/73.27** → A6 **16.98/71.45**, paired ΔMAE **−1.19**, CI **[−2.34,−0.307]**; trên M0-positive thêm **487 box/68 query**, TP **+371**, FP **+116**. | Có tín hiệu nonzero recovery, nhưng không đủ bền để thành all-query solution. [Xác nhận](target_aware_counting/output/a6_consensus_confirm_val_20261102_v1/RESULT.md). |
| **A6 category stress** | 100 FSC VAL khác, loại `polka dot`/`bottle cap` trước inference | A4 **7.35/14.10** → A6 **7.33/14.16** MAE/RMSE; chỉ **+7 TP/+15 FP** từ 22 box M0-positive. CI ΔMAE **[−0.13,+0.09]**. | Gain trước đó tập trung category; giữ A6 làm hypothesis/recall source, chưa promote. [Báo cáo](target_aware_counting/output/a6_category_stress_val_20261102_v1/RESULT.md). |
| **A7** bắt hai tile đồng thuận khi zero | Posthoc replay 73 FSC M0-zero + 98 LVIS absent M0-zero | LVIS activation **1/98 → 0/98** nhưng FSC false-zero **19 → 39**, TP **1599 → 386**, zero-subset MAE **52.51 → 68.86**. | Bác bỏ gate đồng thuận cứng; một tile vẫn có thể thấy vật đúng. [Báo cáo](target_aware_counting/output/a7_zero_consensus_diagnostic_20261102_v1/RESULT.md). |
| **A8** weak-singleton gate | 100 FSC + 100 LVIS absent prospective | A8=A4 trên FSC: MAE **22.42**, TP/FP **4816/415**; gate chặn **0**. LVIS absent: cả A4/A8 activation **1/97**, mass **1**. | Không có incremental value. [FSC](target_aware_counting/output/a8_weak_singleton_val_20261102_v1/RESULT.md), [absent](target_aware_counting/output/a8_weak_singleton_lvis_confirm_20261102_v1/RESULT.md). |
| **A9** cascade thêm chín tile nhỏ nếu A4 vẫn zero | FSC pilot100 + confirm86 source khác | Pooled186: A4 MAE/RMSE **23.968/108.167** → A9 **23.011/107.456**; TP **10773 → 10933**, FP **998 → 1016**, false-zero **5 → 2**. Chỉ **3/186** prediction thay đổi; ΔMAE CI **[−2.774,0]**. | Cứu hiếm; safety chưa đạt. [Xác nhận](target_aware_counting/output/a9_fine_tile_cascade_confirm_val_20261103_v1/RESULT.md). |
| **A9 absent safety** | 100 LVIS verified-absent, M0=0 ở 97 | A4 new-positive **3/97**, mass **27**; A9 **8/97**, mass **43**. | Không promote raw fine cascade. [Báo cáo](target_aware_counting/output/a9_lvis_absent_safety_20261103_v1/RESULT.md). |
| **A10** fine-tile coverage ≥5/9 | Prospective LVIS present100 + absent100 | Gate chặn **mọi** fine activation; A10=A4. Present MAE A4 **1.99**, A9 **1.83**; absent mass A4/A10 **207**, A9 **213**. | Bác bỏ coverage gate; nó bỏ gain thật, không sửa lỗi A4. [Báo cáo](target_aware_counting/output/a10_fine_tile_coverage_lvis_20261103_v1/RESULT.md). |
| **A11** CountGD++ re-detection ở candidate | Development diagnostic trên cùng LVIS A10 | Present MAE A9 **1.83** → A11 **1.87**, TP **107 → 103**; absent mass **213 → 132**, nhưng ca `avocado` trên ảnh banana còn **116** box sai; thêm **264 forward**. | Không phải xác nhận ngữ nghĩa độc lập và quá tốn; không promote. [Báo cáo](target_aware_counting/output/a11_candidate_verification_lvis_20261103_v1/RESULT.md). |
| **A12–A13** SigLIP2 score/COCO80 competition | Diagnostic trên 264 proposal đã biết nhãn | A12 target-only AUC phân biệt target-IoU50/absent **0.673** vs native **0.482**, nhưng `avocado` vẫn score cao. A13 margin AUC **0.737** (CI của gain so A12 **[−0.017,+0.125]**); absent còn **105** mass, trong đó **103** từ một ca. | Có thông tin ranking, chưa có gate đủ an toàn; COCO80 là closed bank hạn chế open-world. [A12](target_aware_counting/output/a12_siglip_a11_semantic_probe_20261103_v1/RESULT.md), [A13](target_aware_counting/output/a13_coco80_competitor_probe_20261103_v1/RESULT.md). |
| **A14** gate A13 kiểm tra prospective | LVIS source-disjoint present100 + explicit-absent100 | Present MAE A4/A9/A14 **1.55/1.56/1.72**; TP50 **121/125/103**; absent positive/mass **1/1**, **6/6**, **1/1**. A14 bỏ **22 TP** so A9, gồm target `magnet` bị competitor COCO lấn. | Safety tốt hơn A9 nhưng mất target, không promote. [Báo cáo](target_aware_counting/output/a14_coco_competition_confirm_lvis_20261103_v1/RESULT.md). |

**Ý nghĩa chung của A:** Quy tắc M0=0 và crop native của A4 là thành phần đáng giữ. Cải thiện lớn hơn cần xét M0-positive, nhưng A5/A6 cho thấy không thể cộng box vô điều kiện. A9 cho thấy scale vẫn là nguồn recall, nhưng không thể đánh đổi absent safety. A12–A14 chứng minh dùng một score/competitor bank cứng dễ loại nhầm target ngoài danh sách.

## 3. Nhánh B và phép so trực tiếp A–B

**B0 SAM3 add-only** trên primary100 cũ: M0 → B0 MAE **19.83 → 18.47**, RMSE **93.43 → 93.19**, point TP **4077 → 4228**, FP **478 → 505**, false-zero **14 → 5**. Absent111 mass **6942 → 6945** và positive **111/111**. Đây là recovery có ích, nhưng B0 không thể xóa lỗi M0 đã dương trên absent và tăng FP. [B0 report](target_aware_counting/output/sam3_fsc_transfer_20261102_v1/RESULT.md).

**Paired FSC A8 cohort100, bản SM86 v2 hợp lệ:**

| Metric trên đúng cùng 100 ảnh/query | M0 | A4 | B0 |
|---|---:|---:|---:|
| MAE | 23.43 | 22.42 | **21.90** |
| RMSE | 122.875 | **122.098** | 122.576 |
| Point TP / FP | 4718 / **412** | 4816 / 415 | **4873** / 456 |
| False-zero | 10 | 5 | **3** |
| Exact count | 4 | **5** | 3 |

B0 so với A4: MAE **−0.52**, RMSE **+0.477**, TP **+57**, FP **+41**. Paired bootstrap CI của chênh MAE **[−1.80,+1.05]**, RMSE **[−1.852,+6.303]**, đều cắt 0: **không có winner rõ**. A4 cải thiện **2 query mà B0 không cải thiện**; B0 cải thiện **18 query mà A4 không cải thiện**. Đây là bằng chứng bổ sung nhau, không phải bằng chứng rằng union là method tốt. Union + NMS posthoc có MAE **21.12**, TP/FP **4951/458**, false-zero **1**, nhưng luật được tạo sau khi nhìn paired result nên **chỉ diagnostic, không được gọi kết quả prospective**. [Paired v2](target_aware_counting/output/b0_a8_paired_val_sm86_20261103_v2/RESULT.md).

**Lỗi thực thi quan trọng:** bản paired v1 trên pc-152 dùng CUDA extension sai kiến trúc; `no kernel image is available`, candidate pool hỏng. [v1](target_aware_counting/output/b0_a8_paired_val_20261102_v1/RESULT.md) là **INVALID EXECUTION** và tuyệt đối không được đưa vào so model. Bản v2 đã build SM86 và kiểm tra forward extension trước khi chấm.

## 4. Điều gì vẫn dùng được, điều gì đã bị bác bỏ?

| Giữ làm tài sản nghiên cứu | Không tiếp tục làm hướng chính |
|---|---|
| A4 frozen làm baseline/candidate source cho M0-zero; 300 FSC paired và các LVIS safety cohort. | A1 reseeding, A3 full-image score gate, A8 singleton gate: không kích hoạt hoặc không cải thiện. |
| B0 frozen cùng paired A8 v2, vì nó cứu các query khác A4. | A2 all-reference và A5 all-query add-only: overcount/FP lớn. |
| A6 cache làm phân tích nonzero undercount; category stress là phép thử giới hạn tổng quát hóa. | A7/A10 consensus/coverage cứng: loại nhiều TP thật. |
| A9 cache cho zero residual; A11–A14 làm negative evidence về semantic gating. | A11 self-verification, A14 COCO80 hard gate: không đạt đồng thời present và absent. |
| Manifest, frozen predictions, capture, hashes, evaluator và các lỗi có provenance. | B paired v1 CUDA lỗi: loại hoàn toàn khỏi science result. |

Vì thế **không xóa artifact, không gộp các cohort tùy ý, không chỉnh ngưỡng trên cohort đã xem**. Các thí nghiệm âm giảm đáng kể không gian giải pháp và tránh lặp lại cùng sai lầm.

## 5. Hướng mới nên thử: câu hỏi hẹp, cổng dừng rõ

### 5.1 Câu hỏi trước khi viết method

Trên cùng ảnh/query, candidate của A4 và B0 khác nhau *thật* ở đâu? Với mỗi candidate được thêm: đúng target hay distractor, một vật hay duplicate/mảnh, có phần nhìn thấy đủ để xác minh không? Trong nhóm A4/B0 chỉ một bên đúng, các tín hiệu frozen nào phân biệt được hai bên mà không mất quá nhiều TP? Dùng cache paired A8 v2 và cache A6/A9 để tạo diagnostic; **không dùng GT tại inference và không tuyên bố diagnostic là confirmation**.

### 5.2 Cơ chế ứng viên, chưa được chứng minh

Một **query-conditioned instance-set reconciliation** không huấn luyện: proposal recall từ M0/A4/B0; gom candidate có thể cùng vật bằng vị trí/box/mask/provenance; dùng bằng chứng query của model frozen để quyết định chọn hoặc từ chối từng nhóm; output một box/mask đại diện và lý do chọn, hoặc tập rỗng. A4/SAM3 chỉ được giữ nếu ablation chứng minh có TP độc lập. Không học cross-encoder, adapter, head hay threshold bằng TEST. Phép che vùng/phản thực là diagnostic tùy chọn, **không** mặc nhiên là verifier.

### 5.3 Điều kiện go/no-go

1. **Cổng cache:** Nếu B0 không đóng góp TP độc lập hoặc không có tín hiệu frozen phân biệt được đúng/sai trên paired audit, dừng hybrid. Giữ A4/B0 là component studies.
2. **Cổng method:** Chỉ khi Cổng cache có cơ sở, khóa rule, model hashes, manifest, candidate/provenance và evaluator trên development **mới**. FSC VAL đã được khai thác rộng; không gọi một phép thử thêm trên nguồn FSC cũ là independent confirmation. Không mở TEST.
3. **Cổng confirm:** So M0, A4, B0, union thô, hybrid và baseline gần như SAM3Count trên cùng image/query. Báo paired MAE/RMSE, point + extent/identity, duplicate, exact count, verified-absent new-positive **và count mass**, latency/VRAM, category/density/occlusion bins và CI. Nếu gain chỉ là một category/outlier hoặc tăng absent mass, không promote.

Đây là kế hoạch tìm solution mạnh **có quyền thất bại**. Chưa có bằng chứng để cam kết hybrid sẽ thắng. Nếu các nguồn frozen tiếp tục cùng nhầm target, ràng buộc không huấn luyện có thể là giới hạn thực tế; khi đó cần báo rõ tradeoff thay vì thêm gate hậu nghiệm.

## 6. Nghiên cứu liên quan: ranh giới claim

[SAHI](https://arxiv.org/abs/2202.06934) đã làm slicing inference; [CountGD++](https://arxiv.org/abs/2512.23351) đã có positive/negative prompts và pseudo-exemplars; [CountZES](https://arxiv.org/abs/2512.16415) đã làm training-free exemplar selection; [SAM3Count](https://openreview.net/pdf/42549a61b00ae67574067e1eea5dff27637b4703.pdf) đã ghép SAM3 + ROI tiling cho zero-shot counting; [PrACo++](https://arxiv.org/abs/2605.02752) đã đặt negative-label và distractor tests cho semantic grounding. Do đó không thể claim các thành phần đó là mới. Contribution nếu có phải là **một quyết định instance-set theo query có bằng chứng paired, absent-safe và localization đúng**, được chứng minh so với các đối chứng trên nguồn mới.

## 7. Câu trả lời ngắn cho cuộc họp

> Chúng ta không phải bỏ kết quả cũ. A4 đã chứng minh crop cứu false-zero trên FSC VAL, B0 cứu thêm các query khác A4 trên một phép so paired hợp lệ, và A6 chỉ ra dư địa ở M0-positive. Nhưng các thử nghiệm thêm box hoặc gate score đơn giản đều cho thấy lỗi FP, duplicate và query absent. Hiện chưa có solution final. Việc tiếp theo là dùng cache paired A–B để đo chính xác TP độc lập và lỗi semantic/instance, rồi chỉ khóa một hybrid mới nếu có tín hiệu phân biệt thật; kiểm tra nó trên nguồn mới với present, distractor, occlusion và verified-absent. Không tuyên bố novelty hay mở TEST trước lúc đó.
