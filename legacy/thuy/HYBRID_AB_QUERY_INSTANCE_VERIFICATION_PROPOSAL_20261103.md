# Đề xuất họp: Xác nhận đúng query và đúng instance khi đếm (QIV)

> **Bản này đã được thay bằng [bản họp chốt hướng nghiên cứu](MEETING_FINAL_FROZEN_QUERY_INSTANCE_COUNTING_20261103.md).** Bản mới phản ánh ràng buộc không huấn luyện, các tiền lệ SAM3Count/CountZES/PrACo++ và cổng dừng trước khi xây verifier.

**Ngày:** 03/11/2026  
**Trạng thái:** Đề xuất nghiên cứu, **chưa có kết quả thực nghiệm hybrid**. **Không huấn luyện, fine-tune, tối ưu online hoặc cập nhật trọng số bất kỳ model/head/verifier nào.**  
**Mục tiêu:** Ảnh + query text bất kỳ → tập instance có vị trí, provenance và lý do được chấp nhận; count bằng kích thước tập. Query vắng mặt phải trả tập rỗng.

## 1. Quyết định đề xuất

Thử **Query–Instance Verification (QIV)**: một bước xác nhận riêng quyết định (i) candidate có thỏa **toàn bộ query**, (ii) nó có phải **một instance riêng** hay là bản sao/mảnh của instance khác, và (iii) có đủ bằng chứng để trả kết quả dương khi target có thể vắng mặt. Chỉ tập instance sau bước xác nhận mới được đếm. **A4 và SAM3 là hai nguồn candidate/mask tùy chọn cần kiểm định bằng ablation, không thuộc định nghĩa bắt buộc của QIV.** Điểm khởi đầu tối thiểu là candidate từ CountGD++ ảnh gốc; dùng thêm A4 hoặc SAM3 chỉ khi chúng có giá trị bổ sung trên cùng ảnh/query.

Đây là giả thuyết cần kiểm tra, không phải khẳng định A+B đã thắng. **Không lấy hợp box từ nhiều nguồn làm output.** Không dùng GT count/point/absent ở inference. Không dùng API/VLM bên ngoài trong pipeline chính, không huấn luyện trên FSC hay nguồn khác trong dự án này, không dùng TEST để phát triển. Verifier nếu có chỉ được dùng phép tính cố định trên output/feature của model đã frozen; mọi rule và ngưỡng phải khóa trước confirm.

## 2. Vì sao cần bước xác nhận

- **A4 có ích nhưng cơ chế đơn giản.** Trên hai cohort FSC VAL source-disjoint tổng 300 ảnh, M0 → A4: MAE **17.07 → 14.95**, RMSE **50.28 → 47.37**, point TP **+620**, FP **+21**, false-zero **32 → 7**. A4 chạy bốn crop khi M0 count = 0, dùng cùng query/readout rồi NMS. Đây là positive zero-count recovery; point match chưa xác nhận semantic identity hay box extent.
- **A4 có rủi ro absent.** Trên một cohort LVIS verified-absent 100 query, A4 tạo 7 activation mới và 205 box, có ca sai 181 box. Tập FSC verified-absent111 cũ có M0 dương 111/111, nên không kiểm tra safety của trigger M0=0.
- **B0/SAM3 có tín hiệu bổ sung nhưng add-only chưa an toàn.** Trên FSC primary100 cũ, M0 → B0: MAE **19.83 → 18.47**, point TP **4077 → 4228**, FP **478 → 505**, false-zero **14 → 5**; trên absent111, mass **6942 → 6945**. Hai cohort này không phải phép so A4–B0 trực tiếp.
- **Những gate đã thử không đủ.** A5 thêm tile box mọi query làm FP tăng mạnh; A6 cứu một số nonzero undercount nhưng gain tập trung vài category và không ổn định trên category stress; A7 consensus loại nhiều TP; A11 re-detection bằng CountGD chưa cho bằng chứng độc lập; A12–A14 dùng SigLIP/competitor margin có thể giảm FP nhưng làm mất target thật. Vì vậy không coi “thêm một score/ngưỡng” là solution.
- **Trần cấu trúc của A4:** trên chính pooled FSC300 đã xem, một oracle sửa hoàn hảo mọi query M0=0 nhưng giữ query M0 dương vẫn có MAE **10.83**; khoảng **72%** sai số tuyệt đối còn lại của A4 nằm ở query M0 dương. Đây là bound hậu kiểm trên cohort cũ, không phải dự báo performance.

## 3. Liên hệ với công trình đã công bố

| Cơ chế | Bài học cho dự án |
|---|---|
| [SAHI, ICIP 2022](https://arxiv.org/abs/2202.06934) và adaptive cropping của [CountGD, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/file/57c56985d9afe89bf78a8264c91071aa-Paper-Conference.pdf) | Crop + map box + NMS và crop để tránh giới hạn decoder đã có. Không claim novelty từ A4 đơn lẻ. |
| [RTGen, 2024](https://arxiv.org/abs/2405.19854), [fine-grained OVD benchmark, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Bianchi_The_Devil_is_in_the_Fine-Grained_Details_Evaluating_Open-Vocabulary_Object_CVPR_2024_paper.pdf) | Region–text matching phải phân biệt hard negatives gần giống target, nhất là thuộc tính. |
| [Mask Grounding, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Chng_Mask_Grounding_for_Referring_Image_Segmentation_CVPR_2024_paper.html) | Đặc trưng từ box có thể trộn nền, vật che và nhiều instance; cần kiểm tra vùng/mask và ngữ cảnh. |
| [SAM3, 2025](https://arxiv.org/abs/2511.16719) | Tách concept presence và localization, cung cấp mask/identity proposal; vẫn phải đo lỗi trên query absent và distractor. |
| [CountOCC, 2025](https://arxiv.org/abs/2511.12702) | Che khuất là bài toán suy luận amodal; phần nhìn thấy ít có thể không đủ xác định cả identity lẫn số instance. |
| [Counterfactual Segmentation Reasoning, 2025](https://arxiv.org/abs/2506.21546) | Kiểm tra cả ảnh có target và ảnh đối chứng vắng target; model phải biết từ chối. |
| [Contrastive Region Guidance, ECCV 2024](https://arxiv.org/abs/2403.02325) và [FACTOR, 2026](https://arxiv.org/abs/2605.03294) | Đối chiếu output khi che vùng hoặc biến đổi ảnh ở inference, không cập nhật trọng số. Đây là tiền lệ của kiểm chứng phản thực, không phải novelty riêng của QIV. |
| [CountZES, 2025](https://arxiv.org/abs/2512.16415), [CountingDINO, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Pacini_CountingDINO_A_Training-free_Pipeline_for_Class-Agnostic_Counting_using_Unsupervised_Backbones_WACV_2026_paper.pdf), [SAM3Count, 2026](https://openreview.net/pdf/42549a61b00ae67574067e1eea5dff27637b4703.pdf) | Training-free exemplar selection, feature matching và SAM3 + ROI tiling đã được áp dụng cho counting. Không claim các thành phần này là mới; cần so trực tiếp nếu dùng. |
| [CountGD++, CVPR 2026](https://arxiv.org/abs/2512.23351) | Chính backbone đã hỗ trợ positive/negative text và visual prompts, cùng pseudo-exemplar ở inference. Không claim negative prompting hoặc self-exemplar là mới. |

Các nguồn trên cung cấp cơ chế và đối chứng, **không chứng minh QIV có novelty hoặc hiệu quả**. Riêng tuyên bố “chưa ai áp dụng cho zero-shot counting” hiện không có cơ sở: CountZES, SAM3Count và CountGD++ là tiền lệ trực tiếp cho nhiều thành phần. Muốn claim method cần chỉ ra thành phần cụ thể thực sự khác và so với cùng cơ chế đã có, không chỉ với M0. **Không đề xuất train cross-encoder:** ràng buộc của dự án là inference-only trên model frozen.

## 4. Giả thuyết có thể bác bỏ

**H1 — Semantic:** Candidate đúng target và distractor gần giống có tín hiệu phân biệt ở mức *vùng + ngữ cảnh + query đầy đủ* mà native score đơn lẻ không giữ được. Tín hiệu này phải vượt A4 trên dữ liệu mới mà không làm mất phần lớn TP A4 cứu được.

**H2 — Instance:** Box/mask từ nhiều nguồn có thể được nối thành các nhóm “cùng một vật”, tách các vật chồng lấn và giảm duplicate/merged detections. Nếu mask chỉ giúp đẹp ranh giới nhưng không cải thiện point/instance matching, H2 không được hỗ trợ.

**H3 — Absence:** Presence evidence ở ảnh nguồn kết hợp bằng chứng vùng có thể ngăn các crop tạo count mass sai trên verified-absent có M0=0. Nếu cần loại nhiều true target để đạt điều này, hybrid không đủ tốt.

**H4 — Complementarity:** A4 và/hoặc SAM3 bổ sung candidate TP, mask hoặc bằng chứng identity mà M0 không có trên *cùng image/query*. Nếu lợi ích riêng không vượt chi phí và lỗi tương quan, bỏ thành phần đó khỏi QIV.

## 5. Thiết kế pipeline đề xuất

### 5.1 Candidate generation, giữ provenance

1. Chạy M0 full image như protocol frozen. Đây là nguồn candidate tối thiểu.
2. **Nhánh tùy chọn A4:** chạy đúng luật frozen cho M0=0. Với M0>0, dùng candidate từ artifact A6/B đã có để phân tích trước; mở rộng inference chỉ trong nhánh QIV mới và phải khóa luật riêng.
3. **Nhánh tùy chọn SAM3:** chạy theo cùng canonical query trên đúng image/query được chọn. Box và mask của SAM3 là proposal, không tự động là vật đúng. Nếu mask từ nguồn khác đủ dùng hoặc SAM3 không có giá trị tăng thêm, bỏ SAM3.
4. Mỗi candidate ghi `source`, `query`, `box`, `mask` nếu có, score gốc, crop ID, slot/ID, biến đổi tọa độ, checkpoint/code hash. Bảo toàn mọi output M0 để đối chứng.

### 5.2 Xác nhận query–region

- Kiểm tra danh từ chính và các điều kiện có thể quan sát trong query (màu, bộ phận, trạng thái, quan hệ) ở **mask phần nhìn thấy** và **ngữ cảnh xung quanh** bằng output/feature của model frozen. Không thêm cross-encoder, adapter hoặc head được học. Không tách query bằng danh mục FSC hay dùng category GT.
- So target với các **hard negative hợp lý của cùng query**. Không dùng COCO80 làm competitor bank cố định: A13/A14 đã cho thấy nó có thể loại nhầm target ngoài danh sách. Cách tạo competitor phải cố định trước chấm và không dựa vào nhãn dataset.
- Nếu có visual exemplar tự lấy từ ảnh, chỉ lấy từ candidate có bằng chứng mạnh; phải có nhánh **không có exemplar** khi không có seed đáng tin. So sánh masked appearance chỉ là bằng chứng phụ, vì vật bị che có thể rất khác vật rõ.
- Đánh giá **presence ở ảnh nguồn** độc lập với số box do crop tạo ra. Score từ CountGD++ và SAM3 không mặc nhiên là xác suất đã hiệu chuẩn hay hai nguồn lỗi độc lập.

### 5.3 Phân giải instance

Tạo graph candidate: cạnh “cùng instance” dựa trên giao box/mask, vị trí phần nhìn thấy, đặc trưng và provenance crop. Giải bài toán chọn **một đại diện cho mỗi vật** và cho phép hai mask chồng lấn nếu có bằng chứng hai vật riêng. Không dùng IoU-NMS làm chứng cứ duy nhất. Đối với vùng bị che nhiều, ghi `uncertain`; không suy ra thêm vật chỉ từ hình dạng ẩn được dự đoán.

### 5.4 Output có thể kiểm tra

Mỗi instance được chọn có `candidate_ids`, box/mask, phần query được hỗ trợ, bằng chứng presence/identity, liên kết duplicate đã loại, và `uncertainty`. `count = len(instances)`. Nếu không instance nào đạt luật chấp nhận đã khóa, trả `instances=[]`, `count=0`. Các trường này là provenance và lý do quyết định có thể audit; **không gọi chúng là giải thích nhân quả** nếu chưa kiểm chứng bằng can thiệp/counterfactual.

## 6. Kế hoạch thực nghiệm và điều kiện dừng

### Giai đoạn 0 — Kiểm tra bổ sung A–B bằng cache, không GPU

- Trên **cùng image/query**, lập ma trận A đúng/B sai, B đúng/A sai, cả hai đúng/sai ở mức candidate/instance. Tách M0=0, M0>0, absent, occluded, dense và distractor. Không suy superiority từ các cohort khác nhau.
- Review semantic identity/duplicate **GT-blind trước**, sau đó mới dùng GT point/box/mask để chấm. Coder tự thực hiện review và ghi `unknown` khi bằng chứng không đủ.
- Nếu B không có phần TP độc lập đáng kể hoặc sai cùng A trên absent/distractor, dừng QIV trước khi thêm inference nặng.

### Giai đoạn 1 — Benchmark candidate và verifier, luật khóa trước khi chấm

- Dùng nguồn development **mới**; FSC VAL đã được dùng rộng ở A1–A14, nên các cohort FSC cũ chỉ dùng để phân tích lỗi, không làm confirm cho luật chọn trên chúng. Không mở TEST trong vòng phát triển.
- So sánh M0, A4, B0, union thô A+B (đối chứng lỗi), **M0+QIV**, **M0+A4+QIV**, **M0+SAM3+QIV**, **M0+A4+SAM3+QIV** trên cùng image/query. Chỉ giữ nguồn thêm nếu ablation chứng minh lợi ích. Thêm đối chứng tiled inference/SAHI với ngân sách model forwards tương đương.
- Lưu `input_lock`, code/checkpoint/config hashes, frozen candidate/prediction files trước khi mở GT; dùng source-disjoint và nếu có thể category-disjoint confirm. Mọi ngưỡng/logic phát sinh từ development phải khóa trước confirm.

### Giai đoạn 2 — Chấm các mục tiêu không thể thay thế nhau

**Primary:** paired per-source MAE/RMSE; point TP/FP/FN; exact count; verified-absent new-positive rate và count mass (đặc biệt M0=0). Báo paired bootstrap CI và denominator.

**Identity/instance:** precision của box thêm, duplicate rate, split/merge error, matching bằng mask/extent nếu dataset có annotation thích hợp; tách occlusion/density/attribute/relation bins. Point containment **không** chứng minh đúng semantic identity hay extent.

**Chi phí:** latency theo source, số forwards, VRAM đỉnh; báo riêng chi phí của SAM3 và verifier. Không chấp nhận cải thiện rất nhỏ với chi phí không tương xứng mà không nói rõ.

**Quy tắc quyết định:** chỉ đề xuất QIV làm nhánh final nếu cải thiện có lặp lại trên confirm mới, giữ hoặc cải thiện localization và không tạo thêm absent activation/count mass đáng kể so với A4. Nếu chỉ thắng MAE nhờ một outlier, mất TP nhiều hoặc dồn FP vào absent, bác bỏ phiên bản đó. Mức sai số chấp nhận và phép kiểm định cụ thể phải đăng ký trước confirm, không chọn sau khi xem GT.

## 7. Ranh giới claim và vận hành

- Hiện chỉ claim được A4 là **zero-count recovery component trên FSC VAL**; B0 là đối chứng SAM3 riêng. **Chưa có kết quả QIV/hybrid**. Không gọi A4 crop+NMS là novelty.
- QIV có thể là contribution nếu chứng minh được **query–instance verification + duplicate/occlusion resolution + abstention** vượt các đối chứng trên nguồn mới. Việc dùng A4/SAM3 chỉ là lựa chọn triển khai được giữ khi ablation chứng minh cần thiết. Không gộp kết quả khác cohort hoặc gắn tên hybrid cho union chưa đo.
- Không dùng GT trong inference, không suy verified-absent từ thiếu annotation FSC, không tune `.23`, `.65`, NMS `.5` của A4 trên cohort đã xem. Không dùng TEST trước khi khóa method/evaluator và hoàn tất kiểm tra development ngoài FSC.
- GPU dùng chung: không ảnh hưởng tiến trình người khác. Với lệnh GPU có khả năng chạy trên 3 phút, chuẩn bị lệnh resumable và đưa người dùng chạy; không tự chạy job dài. Ưu tiên phân tích cache CPU ở Giai đoạn 0.

## 8. Câu chốt cho cuộc họp

> A4 đã chứng minh rằng thay đổi góc nhìn cứu được một số false-zero, nhưng crop và NMS đã là kỹ thuật quen thuộc. Nút thắt còn lại là quyết định **vùng nào thật sự thỏa query và mỗi vật được đếm đúng một lần**, kể cả khi chồng lấn hoặc target vắng mặt. Chúng ta đề xuất thử QIV: xác nhận từng instance và cho phép trả rỗng. A4 và SAM3 chỉ được giữ nếu ablation chứng minh chúng bổ sung candidate hoặc bằng chứng hữu ích. Đây là giả thuyết cần bác bỏ hoặc xác nhận bằng paired evaluation và verified-absent trên nguồn mới; hiện chưa có số liệu QIV để claim thành công.
