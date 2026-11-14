# Bản họp chốt hướng nghiên cứu: đếm đúng vật được hỏi bằng các model frozen

**Ngày:** 03/11/2026  
**Trạng thái:** Quyết định về *hướng thử và phép kiểm chứng*; chưa có phương pháp mới được chứng minh, chưa có số liệu hybrid A+B.  
**Ràng buộc cứng:** Không huấn luyện/fine-tune/adapter/head/verifier, kể cả trên dữ liệu khác. Không tối ưu trọng số ở test time. Không dùng GT count, point, box hoặc nhãn absent trong inference. Không dùng API/VLM bên ngoài trong pipeline chính. Không dùng FSC TEST trong phát triển. Tôn trọng GPU dùng chung; job có thể quá 3 phút phải chuẩn bị lệnh để người dùng chạy.

## 1. Quyết định để nhóm thảo luận

**Không tiếp tục coi A4 crop + NMS là contribution chính; không ghép hộp A+B rồi gọi đó là hybrid.** Đề xuất một vòng nghiên cứu có điều kiện: **Frozen Query–Instance Evidence (FQIE)** — dùng các model đã huấn luyện sẵn ở chế độ frozen để xét *từng candidate*, sau đó chọn một tập các instance thỏa query, không trùng nhau và có quyền trả rỗng. A4 và SAM3 chỉ là nguồn candidate/mask **tùy chọn**. Vòng đầu là feasibility audit không GPU từ cache hiện có; chỉ triển khai inference mới khi audit cho thấy bằng chứng thực sự phân biệt được target với distractor.

**Câu hỏi nghiên cứu:** Với image + query text bất kỳ, có thể dùng *bằng chứng sẵn có từ các model frozen* để quyết định vùng nào là vật được hỏi, mỗi vật được đếm đúng một lần, và trả 0 khi target thật vắng mặt — tốt hơn M0/A4/B0 trên cùng ảnh/query hay không?

Đây là giả thuyết có thể bác bỏ. Nếu không vượt được các đối chứng trên nguồn mới, kết luận đúng là **giới hạn của frozen inference**, không thêm ngưỡng và không tự đặt tên method final.

## 2. Target thật, cách chấm và phần đã đo

Output yêu cầu là **tập instance có provenance**, count bằng số phần tử. Mỗi phần tử phải chỉ đến vị trí trong ảnh và hỗ trợ được query; target vắng mặt phải cho tập rỗng. Đo đồng thời count MAE/RMSE, point/box/mask localization, semantic identity, duplicate/split/merge, và absent activation/count mass. Point nằm trong box chỉ là bằng chứng hình học, **chưa** chứng minh đúng identity hay extent.

| Kết quả đã khóa | Bằng chứng | Giới hạn |
|---|---|---|
| A4 trên FSC VAL 300 source-disjoint: M0 → A4 MAE **17.07 → 14.95**, RMSE **50.28 → 47.37**, point TP **+620**, FP **+21**, false-zero **32 → 7** | Crop giúp phục hồi một số query M0=0 có vật thật. | Luật chỉ can thiệp M0=0; crop + NMS không mới; identity/extent chưa xác minh đầy đủ. |
| B0 SAM3 trên FSC primary100 cũ: MAE **19.83 → 18.47**, TP **4077 → 4228**, FP **478 → 505**, false-zero **14 → 5** | SAM3 có một số positive additions. | Add-only tăng FP; cohort này **không** phải so A4–B0 trực tiếp. |
| A4 trên LVIS verified-absent100 từng đo: **7 activation mới / 205 box mới**; có một ca sai **181 box** | Trigger M0=0 không bảo đảm target có mặt. | Không thể dùng FSC absent111 cũ để kiểm tra trigger này: M0 dương 111/111. |
| A5–A14 | All-query add-only, consensus, re-detection, SigLIP/competitor gates đã thử. | A5/A6 tăng FP hoặc gain không đều; A7 mất nhiều TP; A12–A14 giảm FP nhưng loại nhầm target. Không có verifier đáng tin đã được chứng minh. |

Trên pooled FSC300 đã xem, oracle sửa hoàn hảo **chỉ** các query M0=0 nhưng giữ query M0 dương vẫn có MAE **10.83**; khoảng **72%** sai số tuyệt đối còn lại của A4 thuộc query M0 dương. Đây là bound hậu kiểm của một cohort, không phải dự báo hiệu năng. Để làm solution mạnh phải xử lý cả *nonzero undercount* và *query absent*.

## 3. Tổng quan nghiên cứu và novelty boundary

| Tiền lệ chính | Ý nghĩa đối với đề xuất |
|---|---|
| [SAHI, ICIP 2022](https://arxiv.org/abs/2202.06934); [CountGD, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/file/57c56985d9afe89bf78a8264c91071aa-Paper-Conference.pdf) | Crop, ánh xạ box, gộp prediction và adaptive cropping đã có. A4 không thể claim novelty từ thao tác này. |
| [CountGD++, CVPR 2026](https://arxiv.org/abs/2512.23351) | Model đã hỗ trợ **positive/negative text và visual prompts** cùng pseudo-exemplar. Không claim negative prompting hay self-exemplar là mới. M0 của dự án đang dùng pseudo-exemplar = 0. |
| [CountZES, 2025](https://arxiv.org/abs/2512.16415); [CountingDINO, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Pacini_CountingDINO_A_Training-free_Pipeline_for_Class-Agnostic_Counting_using_Unsupervised_Backbones_WACV_2026_paper.pdf) | Chọn exemplar và matching đặc trưng để đếm **không huấn luyện** đã có; không claim hướng này mới. |
| [SAM3Count, 2026](https://openreview.net/pdf/42549a61b00ae67574067e1eea5dff27637b4703.pdf) | SAM3 + ROI tiling + deduplication **đã dùng cho zero-shot counting**. Đây là đối chứng gần, cần so khi xét hybrid; không claim “chưa ai ghép SAM3 với counting”. |
| [UniCounting, 2026](https://arxiv.org/abs/2610.08379) | Proposal consolidation nhằm xử lý duplicate/fragment và absent-class mass đã có ở bài toán **image-query-free fixed-vocabulary multi-category**; paper này có learned relation, không đúng ràng buộc của ta, nhưng là tiền lệ cấu trúc cần ghi. |
| [Contrastive Region Guidance, ECCV 2024](https://arxiv.org/abs/2403.02325); [FACTOR, 2026](https://arxiv.org/abs/2605.03294) | So output trước/sau can thiệp vùng/appearance khi inference, không học thêm trọng số, đã có ở grounding/OVD. Phản thực là phép **đo bằng chứng**, không tự thành verifier hay novelty. |
| [Does it Really Count?/PrACo++, 2026](https://arxiv.org/abs/2605.02752) | Negative-label và distractor tests trực tiếp phơi bày lỗi semantic grounding ở text-guided counting. Đây nên là phần protocol đánh giá bắt buộc. |

**Kết luận novelty:** Tìm kiếm hiện tại **không đủ cơ sở để nói “chưa ai áp dụng cơ chế này cho zero-shot object counting”**. Nhiều thành phần đã xuất hiện trong counting, kể cả training-free. Khoảng trống *có thể* còn nằm ở một phép quyết định **query-specific, instance-level, có abstention, dùng bằng chứng frozen và được kiểm tra trên positive/absent/distractor cùng nguồn**. Đây là *giả thuyết về khoảng trống*, không phải claim novelty. Phải làm related-work audit đầy đủ trước submission.

## 4. Cơ chế FQIE cụ thể — không có model mới được học

### 4.1 Candidate là recall, chưa phải count

Lấy candidate từ M0 trước. Thử thêm A4 với M0=0, SAM3, và nếu cần nhánh nonzero được khóa riêng. Mỗi candidate giữ query, box/mask, score gốc, source, crop ID/slot, tọa độ ảnh nguồn và code/checkpoint hash. **Union chỉ là tập vùng để xét.** Ablation quyết định A4/SAM3 có cần thiết không; FQIE không phụ thuộc vào tên hai nhánh.

### 4.2 Bằng chứng ngữ nghĩa dùng model frozen

Đối với candidate, kiểm tra query trên **vùng vật nhìn thấy** và **ngữ cảnh** bằng output/feature của CountGD++ và/hoặc SAM3 đã frozen. Tránh embedding của cả box khi box chứa nền, vật che hoặc nhiều vật. Nếu dùng negative text/visual prompt của CountGD++, đó là năng lực sẵn có của model; negative chỉ được sinh bằng rule không dựa GT/category của benchmark. Không dùng COCO80 cố định làm bank phủ quyết: A13/A14 đã loại mất target thật. Không xem score của hai model là xác suất đã hiệu chuẩn hay bằng chứng độc lập.

Kiểm tra phản thực (mask vật/giữ vật/đổi ngữ cảnh) **chỉ là một diagnostic tùy chọn**: dự đoán thay đổi vì có thể tạo ảnh ngoài phân phối, và khi ảnh có nhiều target thì presence toàn ảnh không nhất thiết đổi. Chỉ giữ tín hiệu này nếu audit chứng minh nó phân biệt TP/FP tốt hơn score gốc, không làm hại vật bị che. Không đặt một ngưỡng chỉ vì tín hiệu nghe hợp lý.

### 4.3 Bằng chứng một vật hay nhiều vật

Liên kết các candidate có thể là **cùng instance** qua tọa độ, mask giao nhau, phần nhìn thấy, kích thước và provenance từ các crop. Cho phép hai mask chồng nhau nếu có ranh giới hai vật riêng; gom các mảnh/box của cùng vật. Không mặc định mask đẹp nghĩa là đúng query, và không đếm thêm vật chỉ từ phần hình dạng ẩn do model tưởng tượng. Phần quá che khuất phải có trạng thái không chắc chắn; xác định rõ policy trước khi chấm.

### 4.4 Quyết định và output

Chọn tập instance có bằng chứng query và ranh giới riêng; count = kích thước tập. Nếu không candidate nào đủ bằng chứng, output `[]` và count 0. Mỗi instance lưu candidate IDs, box/mask, bằng chứng ngữ nghĩa, duplicate đã gom và nguồn model để audit. Đây là provenance kiểm tra được, **không phải giải thích nhân quả đã được chứng minh**.

**Phần chưa biết:** Hiện chưa có luật kết hợp score đủ đáng tin. Luật phải được xây từ *diagnostic khả năng phân biệt trên development*, giữ cố định trên confirmation mới. Nếu không tồn tại tín hiệu phân biệt đủ rõ, không cố viết luật từ GT của cohort đã xem.

## 5. Kế hoạch kiểm chứng theo cổng quyết định

### Cổng 0 — Audit tính khả thi, ưu tiên CPU/cache hiện có

Trên **cùng image/query**, phân loại candidate của A và B thành đúng target, sai target, duplicate, split/merge, occluded, unknown. Review identity **không nhìn GT trước**; sau khi khóa nhãn review mới dùng GT để đo. Coder tự thực hiện, không yêu cầu người dùng gán nhãn thủ công. Tính ma trận A đúng/B sai, B đúng/A sai, cả hai sai; tách M0=0, M0>0 và verified-absent. Đo xem native score, SAM3 presence/mask, negative prompt, masked feature và phản thực có thực sự tách các nhóm không. Đây là **phân tích chẩn đoán**, không phải phương pháp đã được chấm confirm.

**Dừng sớm** nếu B gần như không có TP riêng hoặc các tín hiệu đều chồng lấn mạnh giữa TP/FP/absent. Khi ấy hybrid frozen không có cơ sở.

### Cổng 1 — Khóa inference rule và nguồn development mới

Chỉ nếu Cổng 0 có tín hiệu hữu ích: đặc tả một phiên bản FQIE tối giản. Khóa candidate sources, transformations, feature extraction, rule chấp nhận/gom, thresholds và tie-break trước mở GT của confirm. Lưu manifest/hash, frozen candidate và prediction; đánh giá trên nguồn **khác những FSC VAL đã dùng để chọn luật**. FSC VAL đã được khai thác rộng ở A1–A14; không gọi reuse cohort là independent confirm. Không dùng TEST.

### Cổng 2 — So cùng ảnh/query với đối chứng mạnh

Đối chứng tối thiểu: M0, A4, B0, union thô A+B, FQIE chỉ M0, FQIE + A4, FQIE + SAM3, FQIE + cả hai; thêm [SAM3Count](https://openreview.net/pdf/42549a61b00ae67574067e1eea5dff27637b4703.pdf) và training-free exemplar baseline ở setting tương thích nếu triển khai được. Báo cùng image/query và ngân sách forwards/latency/VRAM. Ablation bỏ từng tín hiệu để biết thứ gì thực sự đóng góp.

**Primary endpoints:** paired per-source MAE/RMSE; point/box/mask TP/FP/FN; verified-absent activation rate và count mass với denominator rõ. **Stress tests:** negative-label, distractor, attribute/relation, crowded, occluded, M0=0 và M0>0. Dùng paired CI; báo outlier influence, exact count và semantic identity. Không gộp cohort khác nhau thành paired result.

**Go/no-go:** Chỉ tiến tới final method nếu trên confirmation mới FQIE cải thiện count và localization có lặp lại, không tăng đáng kể absent risk, và gain không phụ thuộc một outlier/category. Mức chấp nhận cụ thể và phép thống kê phải đăng ký **trước** confirm. Nếu không đạt, giữ A4/B0 là các component/đối chứng riêng và báo negative result.

## 6. Thứ không được nói trong cuộc họp

- “Hybrid A+B đã tốt hơn A4” — **chưa đo/khóa như một method thống nhất**.
- “SAM3 xác nhận đúng semantic identity” — mask/presence của một model có thể sai; B0 tăng FP.
- “Crop, SAM3, self-exemplar, negative prompt hoặc phản thực là novelty chưa ai làm cho zero-shot counting” — tài liệu trên cho thấy nhiều tiền lệ trực tiếp/gần.
- “Point TP chứng minh box đúng vật” — point containment không đo đầy đủ identity/extent.
- “FSC absent111 chứng minh A4 an toàn” — M0 dương 111/111 nên A4 không kích hoạt.
- “Không huấn luyện nghĩa là không có model” — CountGD++, SAM3 và các encoder frozen vẫn là pretrained models; chỉ không cập nhật chúng.

## 7. Câu chốt 60 giây

> Mục tiêu là đếm đúng **những instance thỏa query**, có vị trí kiểm tra được, và trả 0 khi query vắng mặt. A4 đã cứu một số false-zero nhưng chỉ là crop fallback; B0/SAM3 có thêm TP nhưng cũng thêm FP. Các công trình mới đã làm SAM3 + tiling, self-exemplar và training-free counting, nên ta không claim chúng là mới. Chúng ta đề xuất một vòng thử có cổng dừng: trước hết dùng cache để xem bằng chứng từ các model frozen có phân biệt được đúng vật, distractor và duplicate trên cùng ảnh/query không. Chỉ khi có tín hiệu thật mới khóa một quy tắc chọn tập instance, đánh giá trên nguồn mới với present, absent, occlusion và distractor. Nếu không vượt các đối chứng, ta công bố giới hạn của hướng frozen thay vì hứa một solution chưa có.
