# Hướng nghiên cứu tiếp theo: chọn tập instance theo toàn bộ query

**Ngày viết:** 03/11/2026  
**Trạng thái:** đề xuất nghiên cứu có cổng dừng; **chưa phải method cuối, chưa có kết quả thực nghiệm của pipeline đề xuất**.  
**Phạm vi:** ảnh + text query bất kỳ → tập instance vật lý thỏa query; `count = |tập instance|`; query đã xác minh vắng mặt phải có khả năng trả `[]` và 0. Không học thêm tham số, không fine-tune, không test-time optimization, không dùng API/VLM bên ngoài trong pipeline chính, không dùng GT/nhãn absent ở inference, không mở FSC TEST để phát triển.

Tài liệu này trả lời câu hỏi: **với những tài sản nghiên cứu hiện tại, làm thế nào đi tới một ứng viên solution cuối có thể bị bác bỏ bằng thí nghiệm?** Nó thay thế ý định triển khai full [FQIE v2](FQIE_V2_RESEARCH_IMPLEMENTATION_20261103.md) ngay lập tức bằng các phép kiểm tra **độc lập theo thành phần**. Cache kiểm coverage và tín hiệu đã lưu; một phép thử nhỏ có đối chứng được phép tạo evidence mới sau khi adapter/checkpoint qua smoke kỹ thuật. FQIE v2 vẫn là một nhánh **remove-only semantic filter** có thể thử như đối chứng; không được đồng nhất nó với lời giải đầy đủ cho identity, nonzero undercount hay query có thuộc tính/quan hệ.

## 1. Điều đã biết và điều chưa được chứng minh

Đọc số liệu cùng cohort, không so MAE chéo cohort. [Bản danh mục A1–A14/B0](EXPERIMENT_PORTFOLIO_DECISION_20261103.md) lưu chi tiết provenance và các kết quả âm.

| Quan sát đã có | Hệ quả cho thiết kế |
|---|---|
| A4, FSC VAL pooled300: MAE 17.07→14.95; TP điểm +620; false-zero 32→7. | Giữ A4 frozen làm nguồn phục hồi khi M0=0. Không gọi crop/NMS là đóng góp mới. |
| A4 chỉ chạy khi M0=0; trên cohort đã xem, phần lớn sai số còn lại nằm ở M0-positive. | Chỉ sửa nhánh zero không thể là solution chung. Cần đo trần recall của candidate ở M0-positive. |
| B0 thêm TP trên một số query khác A4, nhưng tăng FP; trên paired FSC A8-v2 B0−A4 có CI MAE/RMSE cắt 0. | B0 là nguồn proposal tiềm năng, chưa phải verifier hay winner. Cần audit *từng instance* trước khi union. |
| A5 union add-only tăng FP mạnh; A7/A10 gate đồng thuận cứng bỏ nhiều TP; A14 competitor gate mất 22 TP50 trên present100. | Không dùng union/NMS hoặc một score/consensus làm quyết định cuối khi chưa có evidence phân biệt target và distinct instance. |
| FSC absent111 cũ: M0 dương 111/111, mass 6942. Một số LVIS absent cohort cho thấy A4 có thể tạo activation khi M0=0. | Phải kiểm tra cả **xóa lỗi M0 dương khi absent** lẫn **không tạo positive mới khi phục hồi**; không dùng một cohort để thay cho câu hỏi của cohort kia. |
| CycleGRPO smoke cho caption vùng có tín hiệu, nhưng ownership YES/NO sai ở vật bên cạnh; RC0.1 thiếu reference/competitor và matcher không hội tụ. | Caption, cycle và reference matching chưa là oracle. Bắt buộc có capability gate ngắn trước job dài. |

Những kết quả trên là development. Point nằm trong box là matching hình học, chưa chứng minh semantic identity, extent hay một box tương ứng đúng một vật. Việc B0 cải thiện 18 query mà A4 không cải thiện trong [paired report](target_aware_counting/output/b0_a8_paired_val_sm86_20261103_v2/RESULT.md) cũng chưa cho biết có bao nhiêu **GT instance độc lập** có thể lấy thêm mà không đếm trùng.

### Giới hạn bất khả tránh

Một bộ chọn chỉ lấy từ candidate có sẵn không thể tìm vật không nằm trong bất kỳ candidate nào. Một bộ lọc remove-only không thể tăng số TP matching tối đa của base. Khi hai model cùng nhầm distractor, việc chúng đồng ý không chứng minh target đúng. Khi ảnh không cho thấy đủ thuộc tính hay ranh giới, một hệ thống frozen có thể không có evidence để quyết định. Đây là các điều kiện dừng thực nghiệm, không phải lý do để điền một module identity bằng luật đoán.

## 2. Đơn vị đầu ra và các câu hỏi phải giải riêng

Một prediction cuối có cấu trúc tối thiểu:

```text
image_id, query_id, instance_id, representative_candidate_id,
source_candidate_ids, box_or_mask, evidence_for_query,
evidence_for_distinct_instance, uncertainty_flags
```

`count` bằng đúng số `instance_id` được chấp nhận. Mỗi instance phải trỏ tới vùng nhìn thấy và cho phép kiểm tra bằng ảnh/annotation. Một lời giải thích bằng văn xuôi do VLM tạo ra không thay thế được các trường evidence này. Candidate bị từ chối cũng phải lưu ID, vùng và lý do. `[]` là một output hợp lệ; chỉ gọi đó là quyết định absent đúng khi có nhãn verified-absent trong **evaluation**, không suy từ tập proposal rỗng.

Tách bốn câu hỏi, tránh để một score trả lời cả bốn:

1. **Query có nghĩa gì?** Giữ noun, thuộc tính, trạng thái, quan hệ, lượng từ và phủ định nếu có. `apple`, `red apple`, `whole apple`, `apple on the table` tạo bốn predicate khác nhau. Một parser không hiểu facet phải đánh dấu `UNSUPPORTED/UNKNOWN`; không âm thầm hạ query về head noun rồi tuyên bố đã kiểm tra toàn query.
2. **Vùng evidence thuộc về vật nào?** Box có thể bao nền, vật che và nhiều vật. Candidate mask chỉ là giả thuyết vùng vật; mask được prompt bằng query có thể mang cùng sai lệch với detector. Phải đo chất lượng vùng trên case nhiều vật/che khuất, không xem mask đẹp là xác nhận ngữ nghĩa.
3. **Vùng có đúng target không?** Text–image score, descriptor từ caption, SAM3 presence hoặc negative prompt chỉ là tín hiệu frozen. Cần đo lỗi đồng thời và giữ trạng thái `UNKNOWN`. Query-agnostic caption vẫn có phụ thuộc gián tiếp vì proposal ban đầu được query chọn.
4. **Có bao nhiêu vật riêng?** Hai proposal có thể trùng vật, là hai vật sát nhau, hoặc là một mảnh với vật nguyên. Box IoU, mask IoM và cosine không tự quyết định identity. Phải có nhãn kiểm tra độc lập cho các nhóm xung đột, kể cả trường hợp chồng vùng nhưng là hai vật.

## 3. Giả thuyết có thể bác bỏ

> Trên những ảnh/query có candidate coverage đủ, bằng chứng frozen về toàn query và cấu trúc vùng có thể chọn một tập instance từ M0/A4/B0 tốt hơn từng base và union thô, đồng thời không làm tăng verified-absent risk trên nguồn mới.

Giả thuyết này gồm ba mệnh đề **kiểm độc lập**, không phải ba điều kiện cùng phải thắng trước khi được thử một thành phần:

- **H1, candidate complementarity:** A4/B0 cung cấp GT instance chưa được M0 hoặc nguồn kia đại diện, không chỉ thêm box chứa cùng GT point.
- **H2, semantic discrimination:** tín hiệu frozen phân biệt được target với distractor và query absent trên cùng ảnh, kể cả khi query chỉ khác một thuộc tính/quan hệ; lỗi của các tín hiệu không hoàn toàn trùng nhau.
- **H3, instance discrimination:** có tín hiệu đủ để phân biệt duplicate/part với hai vật riêng trong các conflict group thường gặp. Nếu không, chỉ được claim cải thiện count/semantic trên phạm vi đã đo, không claim identity solved.

**Quyết định theo nhánh:** H1 quyết định việc đưa B0 vào candidate source; H2 quyết định có phát triển semantic verifier hay không, có thể thử trên M0/A4 dù H1 thất bại; H3 quyết định có phát triển merge/identity rule hay không, có thể audit trên candidate hiện có dù H1/H2 thất bại. Một nhánh không đạt chỉ đóng claim và thành phần tương ứng. Method cuối chỉ được claim giải đồng thời ba bài toán nếu từng thành phần cần thiết đều qua cổng và kết quả end-to-end đạt trên nguồn mới.

Không được xem H1 đúng chỉ vì MAE union hậu kiểm giảm. Không được xem H2 đúng chỉ vì AUC > 0.5 nếu threshold giữ TP vẫn để absent mass lớn. Không được xem H3 đúng chỉ vì NMS giảm số box.

## 4. Thiết kế ứng viên, chỉ triển khai sau cổng khả năng

### 4.1 Query contract

Lưu nguyên văn query và một bản phân tích deterministic thành các facet: `head object`, `attribute/state`, `relation(anchor, predicate)`, `negation`, `unsupported`. Parser chỉ dùng text query, không dùng GT, nhãn dataset, predicted distractor hay caption ảnh để sửa nghĩa. Những facet không phân tích chắc phải giữ nguyên trong bản text gửi model và được tính trong stress test `unsupported`; không biến chúng thành điều kiện đã được chứng nhận. Không xây closed category bank từ COCO/FSC để phủ quyết query open-world.

Việc dùng parser không phải novelty. Nếu query phức tạp vượt năng lực model frozen, báo giới hạn trên nhóm đó. Evaluation phải tách head-noun, attribute, state, relation và query-switch cùng ảnh; không lấy kết quả FSC class name đơn giản để khẳng định arbitrary text.

### 4.2 Proposal recall

- **M0:** final F2 CountGD++ frozen, giữ nguyên ID/box/score/provenance. F1 raw là nguồn audit riêng, không ghép slot F1 với F2 như cùng prediction.
- **A4:** branch đã khóa, chỉ chạy theo count M0 **trước** mọi semantic filtering. Không gọi lại A4 sau khi xóa box.
- **B0:** SAM3 branch frozen dùng làm nguồn bổ sung **nếu H1 qua cổng**. Mask/presence được lưu làm candidate evidence, không mặc định là xác nhận target. Cần xác minh checkpoint, association rule và CUDA parity như bản paired v2.
- **M0-positive recall:** hiện chưa có nguồn đã qua stress test. A6 cache dùng chẩn đoán bound; không promote A6 vào method chỉ vì gain pooled. Nếu audit cho thấy nhiều GT bị thiếu ở M0-positive và A4/B0 không phủ, thiết kế hiện tại **không thể** đạt mục tiêu chung.

Union ở đây chỉ là một **danh sách proposal để kiểm tra**. Nó không phải tập prediction cuối và không được chấm như method nếu rule được tạo sau khi xem GT. Mỗi source có thể góp TP riêng hoặc chỉ góp nhiễu; ablation sẽ quyết định giữ/bỏ source.

### 4.3 Bằng chứng vùng và ngữ nghĩa

Kiểm tra checkpoint/interface local và khóa revision/hash **trước** smoke kỹ thuật. Sau khi ảnh/vùng/score/cache chạy đúng, được phép tạo một tập evidence mới nhỏ, cố định, có nhãn review và đối chứng trên development để đo năng lực semantic. Cache cũ không thể chứng minh Alpha-CLIP hoặc một prompt CycleGRPO mới có ích khi chúng chưa từng chạy trên các candidate đó. Một candidate có thể có các tín hiệu sau, nhưng không bắt buộc dùng hết:

| Tín hiệu | Câu hỏi hữu ích | Điều nó không chứng minh |
|---|---|---|
| Native CountGD++ query score/token evidence | Candidate được model gốc ưu tiên tới đâu? | Xác suất target đúng; independent verification. |
| SAM3 concept mask/presence | Có vùng/mask cho concept ngắn và dấu hiệu hiện diện? | Đúng toàn query, đặc biệt quan hệ/trạng thái; distinct identity trong pipeline này. |
| Local frozen region caption không nhận query | Vùng quan sát giống vật gì, có nhiều vật/part không? | Caption là nhãn đúng; query independence tuyệt đối. |
| Region–text comparison với alpha/vùng | Vùng có ưu tiên query hơn một mô tả đối chứng cụ thể? | Calibration, entailment logic, hay absence proof. |
| Cross-view/context consistency | Câu trả lời có đổi khi nhìn vùng và ngữ cảnh? | Hai quan sát độc lập thống kê. |

Với Alpha-CLIP, **alpha hình chữ nhật vẫn có thể chứa nhiều vật**; ablation alpha từ mask thực chỉ được thêm khi mask có chất lượng vùng đủ qua audit. Ảnh RGB và alpha phải biến đổi đồng bộ. Descriptor `d` quan sát được chỉ làm đối chứng khi parser gắn được object/facet với referent trong vùng; từ `d ≠ q` không suy ra mâu thuẫn. Noun synonym/hypernym, thuộc tính không nhìn thấy và câu caption mơ hồ phải thành `UNKNOWN`. Score margin phải được so với target-only và native-score controls ở cùng TP retention, không gán tên “confidence”.

Nếu phép thử evidence mới không vượt đối chứng ở operating point giữ target, không đưa tín hiệu ấy vào rule. Một ca sai riêng lẻ chỉ bác bỏ bảo đảm tuyệt đối; không đủ ước lượng accuracy. Không tiếp tục tăng prompt, đổi mask, đổi model rồi đánh giá lại trên cùng tập cố định như thể đó là confirmation; một version khác cần protocol và tập xác nhận khác.

### 4.4 Candidate conflict và chọn tập

Một conflict group là tập proposal có khả năng cạnh tranh cho **cùng vùng vật lý**, sinh từ tọa độ, mask, crop/source provenance và adjacency. Đây là **giả thuyết cần kiểm**, không phải nhãn same-instance. Cặp box giao nhau chưa đủ để gộp; hai vật sát hoặc che nhau vẫn có thể cùng được chọn. Cặp part/whole có thể không giao IoU lớn. Review các trường hợp này trước khi khóa luật.

**Chưa đặc tả decoder; chưa viết graph, optimizer hoặc set selector.** Các hành động có thể cần xét là giữ một proposal, giữ hai proposal khi là hai vật riêng, hoặc chọn rỗng. Đây mới là *không gian quyết định*, không phải thuật toán. Candidate gồm nhiều vật chỉ được tách nếu đã có proposal riêng có localization kiểm được; không tạo box/mask tưởng tượng.

Trước khi coder triển khai bất kỳ decoder nào, phải có một **decision specification** kiểm được bằng ví dụ cụ thể, gồm: (1) danh sách feature và cách tạo/hash từng feature; (2) predicate semantic và same-instance có miền áp dụng, ngưỡng và xử lý invalid/UNKNOWN; (3) thứ tự áp dụng rule, representative và tie-break; (4) fallback cho từng loại unresolved, bao gồm chính xác output box/ID/count; (5) các đối chứng cùng candidate pool, tối thiểu base, union+dedup đã khóa và từng rule riêng; (6) các ca phản ví dụ mà rule phải xử lý và các ca nó được phép để `UNVERIFIED`. Đặc tả được khóa trên development trước khi viết decoder và mở nhãn confirmation. Nếu chưa thể viết đầy đủ sáu mục, chỉ làm diagnostic hoặc một thành phần đã xác định như remove-only; không dùng tên “set reconciliation”.

### 4.5 Absent và query-switch

Absent là bài test chủ động, gồm hai cơ chế khác nhau: (a) M0 đã trả nhiều box sai, cần có khả năng xóa; (b) M0=0 nhưng A4/B0 sinh positive mới, cần chặn. Báo cả activation rate và **count mass**, median/p90/max, ảnh outlier. `Y=[]` khi không candidate nào qua rule; tuy nhiên xóa hết không tự chứng minh query thật vắng mặt. Một bộ lọc có thể đạt absence bằng cách xóa target present, nên luôn báo paired target TP/false-zero trên cùng rule.

Query-switch: giữ ảnh cố định, hỏi target class A, class B cùng hiện diện, thuộc tính A1/A2 nếu được annotation, và verified-absent C. So **ID/vùng được chọn**, không chỉ tổng count. Nếu output cùng tập box cho mọi query, semantic alignment thất bại dù một count tình cờ đúng.

## 5. Cổng 0: audit cache cho coverage và tín hiệu đã có

**Input:** paired A8-v2 hợp lệ cho so A4–B0 trên present; các cache verified-absent FSC/LVIS có provenance hợp lệ cho câu hỏi absence riêng; A6/A9 cache chỉ cho diagnostic riêng. Kiểm hash, source-image overlap, F1/F2, evaluator và missing items trước mọi thống kê. Không ghép absent cohort với paired A8 như thể là cùng ảnh/query. Bản paired v1 CUDA lỗi là invalid và không được nhập. Không trộn metric LVIS IoU50 với FSC point matching.

**Nhãn audit:** chọn manifest khóa trước theo source; dùng paired A8 cho M0-zero/positive, A4-only/B0-only, dense, occlusion và conflict/duplicate; dùng absent cache riêng cho query vắng mặt. Người review xem ảnh, query, candidate vùng và provenance được ẩn tên phương pháp nếu khả thi; gán `target-distinct`, `target-duplicate`, `part`, `distractor`, `multi-object`, `insufficient-visible-evidence`, `uncertain`. Chỉ sau khi khóa nhãn review mới mở GT để đối chiếu. Assistant review một mình là exploratory; semantic/identity claim mạnh cần nhãn độc lập hoặc agreement audit. Không gọi point nonmatch là semantic negative nếu chưa có nhãn.

**Phép tính bắt buộc trên những artifact thực sự có trong cache:**

1. Oracle proposal coverage ở mức **GT instance**, một-một, của M0, A4, B0 và union; tách increment chỉ A4/chỉ B0. Báo cả point proxy FSC và identity/extent khi có annotation phù hợp.
2. Ma trận A4 đúng/B0 sai, B0 đúng/A4 sai, cả hai đúng cùng vật, cả hai đúng hai vật khác, cả hai sai; stratify M0=0/positive, category, density, occlusion.
3. Tần suất duplicate/part/multi-object và sai khác giữa IoU/IoM/provenance với nhãn same-instance. Đo false merge và missed duplicate ở các cutoff được khai báo trước, không tune vô hạn.
4. Trên candidate có nhãn chắc, phân phối của **score đã lưu** giữa target/distractor, target được giữ ở các mức absent mass, lỗi đồng thời giữa các model đã có output. AUC chỉ là phụ; mục tiêu là operating point giữ target. Ghi `NOT_AVAILABLE` cho Alpha-CLIP/CycleGRPO evidence chưa chạy, không suy chúng vô ích từ cache thiếu feature.
5. Cost coverage: tổng candidate thực, max/percentile theo ảnh, số caption/region comparisons nếu xử lý mọi box, latency/VRAM ước tính từ smoke có đo. Không extrapolate từ ảnh thưa sang ảnh 427 box.

**Quyết định từng nhánh:** nếu B0 không có GT instance độc lập đáng kể, không đưa B0 vào method; điều đó không chặn thử semantic verifier trên M0/A4. Nếu score **đã lưu** không tách được target/distractor, không promote score đó; điều đó không chặn một phép thử evidence mới có đối chứng. Nếu overlap/IoM/provenance gộp nhầm vật riêng, không viết merge rule từ chúng; điều đó không chặn semantic-only experiment. Nếu các nguồn hiện có không phủ nonzero undercount quan trọng, ghi rõ trần recall và không claim solution chung từ bộ nguồn này. Chỉ ghi `NO_FEASIBLE_FROZEN_SET_RULE` cho **cấu hình evidence và candidate đã thực sự kiểm**, không cho các model chưa thử.

## 6. Cổng 1: smoke kỹ thuật rồi thử evidence mới có đối chứng

Cổng này **không đòi H1, H2 và H3 đều đạt**. H2 có thể dùng M0/A4 để thử verifier dù B0 không qua H1; H3 có thể kiểm conflict trên candidate đã có dù H2 chưa đạt. Trước mỗi phép thử, ghi riêng câu hỏi, model/checkpoint/prompt/transform, nhãn review, đối chứng, ngân sách và điều kiện dừng. Khóa một bộ ca kỹ thuật nhỏ **không dùng làm ước lượng population**:

- target present rõ, target/distractor cùng ảnh, query verified-absent;
- query cùng head noun nhưng khác màu/trạng thái, và một quan hệ không gian;
- crop chứa nhiều vật, vật bị che, hai vật sát/chồng, một part của vật lớn;
- M0=0, M0>0, ảnh thưa và ảnh dày có candidate count cao.

**Bước 1, smoke kỹ thuật:** checkpoint chính thức tồn tại, hash/revision, adapter ảnh/vùng đúng tọa độ, output deterministic trong setting đã khóa, no-query leakage của descriptor, finite scores, parser `UNKNOWN` đúng, parity M0/A4 khi disable selector, resume và cost trên ảnh dày. Một lỗi tọa độ, sai model, thiếu checkpoint, output hỏng hoặc không thể chạy hết ngân sách đã khóa phải được sửa hoặc chặn trước khi diễn giải nội dung. Lưu raw input/output, không chỉ nhãn pass/fail.

**Bước 2, phép thử evidence mới:** chạy trên một manifest development nhỏ, cố định trước inference, có target, distractor, verified-absent và các ca khó đã nêu; dùng nhãn review có provenance. So cùng candidate/query với native score, target-only comparator, no-op/base và tín hiệu mới đứng riêng, rồi mới xét tổ hợp. Báo coverage, parse `UNKNOWN`, target retention tại các mức absent mass, lỗi theo facet và cost; không lấy một caption đẹp làm bằng chứng. Đây là diagnostic/model selection trên development, chưa phải independent confirmation.

Một phản ví dụ nội dung **chỉ dừng ngay** nếu nó làm hỏng giả định bắt buộc của rule đang đề xuất, ví dụ rule đòi “negative control không bao giờ được chấm dương”, hoặc chứng minh adapter không chỉ đúng vùng. Với claim lợi ích thống kê, một phản ví dụ bác bỏ bảo đảm tuyệt đối nhưng không ước lượng accuracy; phải chấm toàn bộ manifest cố định và so đối chứng. Nếu thiếu Alpha-CLIP hợp lệ, ghi `BLOCKED_COMPARATOR`; nếu caption/parse không đạt tiêu chí đã khai báo, ghi `FAILED_CAPABILITY` cho nhánh đó; không âm thầm thay SigLIP/Qwen và gọi cùng version. Không chạy full benchmark chỉ vì smoke kỹ thuật pass.

## 7. Cổng 2: khóa method trên development

Chỉ khi thành phần tương ứng có evidence dùng được mới chốt một executable specification cho **thành phần đó**. Decoder kết hợp chỉ được bắt đầu sau decision specification sáu mục ở §4.4:

- candidate sources và trigger; query parser/version; model/checkpoint/transform hashes;
- feature extraction, xử lý invalid/UNKNOWN, rule semantic và conflict, representative, tie-break, fallback;
- ngưỡng hữu hạn được đăng ký trước (ví dụ `tau=0`, một danh sách quantile nhỏ và `+∞`), quy tắc chọn trên source-split calibration, cùng controls;
- max candidate/runtime policy. Nếu phải cắt topK, báo rõ coverage loss và kiểm nó trên dense cases; không âm thầm bỏ box;
- input manifest, GT/status nằm ngoài inference, freeze predictions hoàn chỉnh trước evaluation, evaluator và paired bootstrap theo source.

Không gọi threshold chọn bằng nhãn DEV là zero-label calibration. Không chỉnh prompt/model/mask/threshold trên cùng confirmation cohort. Nếu chỉ no-op qua retention gate, báo `NO_INCREMENTAL_VALUE`. Nếu FQIE v2 remove-only là bản duy nhất qua cổng, giữ claim đúng phạm vi *false-positive filtering*, không đổi tên thành set reconciliation.

## 8. Cổng 3: xác nhận độc lập

FSC VAL cũ, kể cả source-disjoint từ những run đã phân tích, vẫn là development của chương trình này. Nguồn xác nhận phải được chọn và khóa **sau khi rule hoàn tất**, với source/image/query không dùng để chọn rule hoặc prompt. Không dùng TEST để sửa method. Nếu nguồn mới chưa có nhãn extent/identity, chỉ claim count/point và absence; không tự biến caption hoặc SAM mask thành GT. MUCCA/PrACo++ có thể là external protocol nếu được dùng đúng setting, không tải và xem trước để tune.

**Cùng ảnh/query**, so tối thiểu: M0, A4, B0, union thô đã khóa, FQIE-v2 remove-only, target-only comparator, proposed set rule, và SAM3Count training-free khi protocol/checkpoint tương thích. SAM3Count có cả tài liệu training-free và checkpoint fine-tuned; phải ghi chính xác setting, không gộp hai kết quả. Ablation bỏ A4, B0, semantic evidence, instance grouping và từng view. Báo latency/forwards/VRAM cùng denominator.

**Primary endpoints, đăng ký trước:** paired count MAE/RMSE, point matching và box/mask IoU khi có GT, semantic identity, duplicate/split/merge, exact count, false-zero, verified-absent activation **và mass**. CI theo source; thêm median per-query, category/density/occlusion bins, leave-one-source-out influence và query-switch. Điểm TP FSC phải gọi point TP, không gọi box/identity TP. Mọi missing item là run chưa hoàn thành, không tính prediction 0.

**Điều kiện promote:** cải thiện count **và** localization/identity trên đối chứng mạnh theo paired analysis; target retention không suy giảm đáng kể theo budget đăng ký; absent mass/activation không tăng theo mức rủi ro đã chốt; kết quả không do một category/outlier; chi phí chấp nhận được. Các mục tiêu số trong FQIE v2 (MAE −10%, RMSE không tăng, giữ ≥99% point TP A4 trên development, absent mass −50%) là target thiết kế, chưa phải forecast hay bằng chứng. Trước confirm phải đặt ngưỡng, metric ưu tiên và cách xử lý trade-off cụ thể; không đổi chúng theo kết quả.

## 9. Bố cục code/artifact khi được phép triển khai

Không sửa frozen artifacts. Tạo một run directory mới theo ngày/version, gồm:

```text
manifest/           input images, query IDs, source grouping, status provenance,
                    cache hashes, overlap report; labels stored separately
audit/              blinded review manifest, label provenance, oracle diagnostics
features/           raw model responses, masks/boxes, transforms, feature hashes
decoder/            chỉ tạo sau decision specification; deterministic rule,
                    invalid/UNKNOWN policy, locked config
predictions/        complete per-item candidate ledger and accepted instance set
evaluation/         per-query results, bootstrap, stress bins, failure gallery
RUNBOOK.md          exact commands, environment, checkpoint hashes, time/VRAM
DECISION.md         gate outcomes and allowed claim
```

Triển khai theo thứ tự: inventory/cache validation → audit CPU cho H1 và tín hiệu đã lưu → smoke kỹ thuật cho model mới → phép thử evidence nhỏ có đối chứng cho H2/H3 tương ứng → decision specification của thành phần qua cổng → parity/unit checks → development feature extraction có resume → freeze rule → run độc lập → evaluation. H1/H2/H3 tiến độc lập; không tạo thư mục `decoder/` với graph/optimizer khi chưa có rule cụ thể. Trước job GPU dài, ghi rõ số call tối đa dựa trên manifest thật, chạy foreground smoke ở cả ảnh dày, chuẩn bị lệnh resume cho người dùng theo quy ước workspace. Không tiếp tục một solver không hội tụ hoặc reference gallery thiếu coverage như RC0.1.

## 10. Đóng góp có thể claim và ranh giới novelty

Nếu đạt cổng xác nhận, claim phù hợp có thể là: **một luật inference frozen chọn tập instance theo query với bằng chứng vùng/provenance, cải thiện đồng thời count, grounding và verified-absent trên nguồn chưa dùng để chọn luật**. Đó vẫn cần đối chiếu related work đầy đủ trước submission. CountGD++ đã có positive/negative prompts và pseudo-exemplars; SAM3 đã có concept mask/presence; SAM3Count đã có tiling/dedup cho counting; Alpha-CLIP đã có region–text features; PrACo++ đã có negative/distractor protocol. Không claim riêng từng thành phần ấy là mới.

Nếu chỉ bộ lọc semantic thắng ở cùng target retention, claim chỉ là cải thiện trade-off của M0/A4; identity và nonzero recall vẫn mở. Nếu set rule không qua cổng, kết luận là **ràng buộc/model/candidate cụ thể hiện tại chưa cung cấp đủ evidence**, không phải chứng minh mọi phương pháp frozen bất khả thi. Không báo một solution cuối chỉ vì MAE giảm khi TP hoặc query adherence xấu đi.

## 11. Việc làm ngay

1. Khóa manifest audit từ paired A8-v2, kiểm cache và source overlap; xác định đúng nhãn absent của từng cohort.
2. Tính proposal oracle và ma trận bổ sung A4/B0 ở mức GT instance; tách point proxy khỏi semantic/identity review. Kết quả H1 quyết định riêng việc dùng B0.
3. Review mù một bộ conflict/semantic cases đã chọn trước; báo lỗi và `UNKNOWN`. Cache chỉ được dùng để đánh giá tín hiệu đã lưu và thiết kế tập thử mới, không phán xét model chưa chạy.
4. Độc lập với H1, kiểm checkpoint/interface và smoke kỹ thuật Alpha-CLIP/CycleGRPO nếu chúng là ứng viên H2; sau đó chạy một manifest evidence nhỏ có đối chứng. H3 được kiểm riêng trên conflict cases, không bắt buộc H2 thành công.
5. Chỉ với thành phần qua cổng, viết decision specification về feature, rule, fallback và controls. Chưa viết graph/optimizer; chưa chạy full benchmark của một decoder chưa xác định.

**Quyết định hiện tại:** `COMPONENT_GATES`. Audit cache cho H1/tín hiệu cũ và thử evidence mới quy mô nhỏ cho H2/H3 được tiến hành theo cổng riêng. Chưa có quyền gọi FQIE v2 hoặc hybrid A4+B0 là final method, và chưa có lý do khoa học để chạy full benchmark của một decoder chưa có decision specification.

## Tài liệu đối chiếu

- Nội bộ: [portfolio A1–A14/B0](EXPERIMENT_PORTFOLIO_DECISION_20261103.md), [paired A4–B0](target_aware_counting/output/b0_a8_paired_val_sm86_20261103_v2/RESULT.md), [FQIE v2](FQIE_V2_RESEARCH_IMPLEMENTATION_20261103.md), [bản họp FQIE](MEETING_FINAL_FROZEN_QUERY_INSTANCE_COUNTING_20261103.md), [CycleGRPO capability failure](target_aware_counting/output/cyclegrpo_capability_probe_20261027_v1/CAPABILITY_FAILURE_DECISION.md).
- Ngoài dự án: [CountGD++](https://arxiv.org/abs/2512.23351), [SAM3](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/), [SAM3Count code/protocol](https://github.com/Joan947/SAM3Count), [Alpha-CLIP](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Alpha-CLIP_A_CLIP_Model_Focusing_on_Wherever_You_Want_CVPR_2024_paper.html), [CycleGRPO](https://github.com/devinxzhang/CycleGRPO), [PrACo++/MUCCA](https://arxiv.org/abs/2605.02752), [UniCounting](https://arxiv.org/abs/2610.08379). UniCounting học relation head và theo bài toán fixed-vocabulary image-query-free, nên là tiền lệ cấu trúc, không phải baseline frozen tương đương.
