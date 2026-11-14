# FQIE v2 — đặc tả nghiên cứu và triển khai CountGD++ target-aware

Ngày 03/11/2026. 
## 1. Quyết định và giới hạn phải hiểu trước khi code

Mục tiêu cuối: ảnh + query → số instance đúng query, cùng vị trí làm bằng chứng; query vắng mặt trả 0. Không huấn luyện bất kỳ tham số mới nào. Không fine-tune hoặc test-time optimization. Dùng model pretrained local ở chế độ frozen. Không dùng nhãn trong inference, không mở TEST để phát triển.

Không có bằng chứng hiện tại cho phép cam kết một giải pháp frozen giải đồng thời semantic, recovery và identity. Bản này chốt một pipeline thực thi với hai năng lực tách được: (i) phục hồi false-zero bằng A4 đã có; (ii) kiểm tra semantic false positive bằng observed-region evidence. Identity/part–whole và nonzero undercount được ghi là giới hạn chưa giải, không giấu sau graph/NMS.

Việc tạo một sơ đồ toàn diện không tương đương giải được mọi khối trong sơ đồ. Không yêu cầu coder tự phát minh identity oracle để hoàn thiện API.

## 2. Bằng chứng dự án hiện có

Các số sau đọc từ bản họp ngày 03/11; coder phải đối chiếu artifact trên server trước đưa vào bảng mới, không chạy lại baseline vô ích.

| Cohort | So sánh đã báo | Diễn giải đúng |
|---|---|---|
| FSC VAL300 source-disjoint của A4 | MAE 17.07→14.95; RMSE 50.28→47.37; point TP +620; false-zero 32→7 | A4 có gain recovery thật, nhưng chỉ xử lý M0=0; crop+NMS không phải novelty |
| FSC primary100 cũ | B0: MAE19.83→18.47; TP4077→4228; FP478→505 | SAM3 thêm target và nhiễu; không cùng cohort với hàng trên |
| LVIS absent100 cũ | A4 tạo 7 activation mới, 205 box | M0=0 không đồng nghĩa absent và cũng không chứng minh present |
| FSC absent111 cũ | M0 dương111/111 | Không kiểm tra được trigger A4 M0=0; là bài test semantic suppression |

F1 raw và F2 final CountGD++ khác forward; không ghép ID slot như thể cùng prediction. Thuật toán mới dùng F2 final để chỉnh output và giữ F1 riêng nếu audit.

Các thất bại trước loại bỏ cách dùng: cosine làm identity, overlap làm owner, Qwen chọn joint slate mặc định, OWLv2/SigLIP hard veto, multi-view agreement làm acceptance, lặp lại tuning trên cohort đã xem.

## 3. Rà soát nguồn và novelty

Nguồn chính thức đã tra cứu ngày 03/11/2026:

1. CountGD++: https://arxiv.org/abs/2512.23351 ; https://github.com/niki-amini-naieni/CountGDPlusPlus — positive/negative prompting và pseudo-exemplar là năng lực gốc. Không claim mới.
2. DAVE: https://openaccess.thecvf.com/content/CVPR2024/html/Pelhan_DAVE_-_A_Detect-and-Verify_Paradigm_for_Low-Shot_Counting_CVPR_2024_paper.html — detect-and-verify đã dùng trong counting. Trang CVF bị 403 khi mở lại; thông tin được đối chiếu kết quả chỉ mục CVF, không giả vờ đã đọc lại toàn PDF.
3. Point Segment and Count: https://openaccess.thecvf.com/content/CVPR2024/html/Huang_Point_Segment_and_Count_A_Generalized_Framework_for_Object_Counting_CVPR_2024_paper.html — phân loại proposal bằng CLIP/region là tiền lệ; không claim region classification mới.
4. CycleGRPO: https://devinxzhang.github.io/CycleGRPO-Page/ ; https://github.com/devinxzhang/CycleGRPO ; https://huggingface.co/XinNUS/CycleGRPO-4B — caption↔grounding là chính cơ chế huấn luyện của model. Tái dùng cycle không phải novelty hoặc independent verifier.
5. Alpha-CLIP: https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Alpha-CLIP_A_CLIP_Model_Focusing_on_Wherever_You_Want_CVPR_2024_paper.html — RGB+alpha cho region recognition, đã được fine-tune trước; frozen trong dự án không có nghĩa chưa từng train.
6. SAM3Count: https://github.com/Joan947/SAM3Count ; https://openreview.net/pdf/42549a61b00ae67574067e1eea5dff27637b4703.pdf — SAM3 counting/tiling là tiền lệ gần. Repo ghi WiCV@CVPR2026 và có thông báo fine-tuning code; cần kiểm tra đúng protocol/checkpoint, không tự coi mọi setting trong bài là training-free.
7. AdaCount: https://muhammad-ibraheem-siddiqui.github.io/AdaCount/ — prototype-guided spatial warping và feature modulation cho SAM3 training-free đã có; không dùng chúng như novelty mới. Trang tác giả ghi arXiv2026, code coming soon ở thời điểm đọc.
8. PrACo++/MUCCA: https://arxiv.org/abs/2605.02752 — negative-label, distractor tests và dữ liệu nhiều category cùng ảnh. Đây là protocol gần mục tiêu của dự án, không phải contribution của ta.

Kết luận: chưa có cơ sở claim “đầu tiên” hoặc “chưa ai làm”. Tra cứu này là khảo sát có mục tiêu, không chứng minh đã phủ mọi paper. Đóng góp tiềm năng chỉ nằm ở quy tắc quyết định và bằng chứng thực nghiệm bổ sung sau khi kiểm soát chất lượng model phụ.

## 4. Sửa các lỗi logic của guide cũ

- Không có mâu thuẫn ≠ có evidence target. KEEP phải ghi KEEP_UNVERIFIED nếu chỉ fallback.
- Caption→ground đúng vị trí ≠ caption đúng class. Model có thể tự nhất quán khi sai.
- WordNet khác nhánh ≠ loại trừ lẫn nhau; dog/pet, cup/container có thể cùng đúng. WordNet dùng chuẩn hóa synonym, không sinh disjoint tự động.
- Negative margin chỉ là relative preference, không là calibrated probability hoặc proof absent.
- Mask từ rectangle là region prompt, không phải instance segmentation. Mask đẹp không xác minh semantic.
- “Hai model đồng ý” không tạo độc lập thống kê. Phải đo lỗi đồng thời.
- Remove-only không thể cứu FN hoặc false-zero. Với cùng bộ box, cardinality và maximum matching TP đều không thể tăng sau khi xóa.
- Pruning không tạo absent activation mới là tính chất toán học, không phải evidence classifier an toàn; nó vẫn có thể xóa target thật.
- Giữ hai vật sát nhau và giải part–owner chưa được chứng minh. Không bật merge/replace trong executable v2.

## 5. Pipeline thực thi v2

### 5.1 Chọn bộ prediction đầu vào trước semantic gate

Cài hai chế độ, không gộp box giữa model:

- M0_only: B = output final của M0.
- A4_base: nếu count M0 ban đầu >0 thì B=M0; nếu M0 ban đầu=0 thì B=output A4 đã khóa. Giữ nguyên toàn bộ rule A4, không tái thiết kế crop/NMS.

Sau khi xóa hết B bởi semantic gate, KHÔNG gọi A4 lại: tránh vòng xóa distractor rồi đưa chính distractor trở lại.

B0/SAM3 được đánh giá như branch độc lập về sau, không union trong v2. Đây là cách tái sử dụng gain cũ mà không giả định đã giải cross-model identity.

### 5.2 Tạo mô tả quan sát, không nhận query

Dùng checkpoint CycleGRPO-4B local đã smoke; khóa revision/hash và native adapter hiện có. Không yêu cầu ownership/joint schema.

Với mỗi box trong B, caption native trên ảnh gốc và region prompt của box. Prompt cố định:

“Describe the visible content of the indicated region. Name the main visible object if identifiable. If the region contains several objects or only a part, state that explicitly.”

Dùng native template/mask tokens đã audit; không đưa query, count, GT, class list hay candidate coordinates vào nội dung mô tả bằng chữ. Region geometry vẫn là input native hợp lệ.

Lưu free text. Parser chỉ chấp nhận noun phrase đơn giản qua lexicon/parser cố định, không dùng chính query để chọn noun trong câu. Part, multiple, ambiguous, empty, truncation, parse failure → UNKNOWN. Không coi unknown là background.

Một caption generic như “object” không sinh negative. Attributes/relations không có cấu trúc phân tích chắc → ngoài scope v2, fallback.

### 5.3 Không sử dụng tự động WordNet disjoint

Descriptor d và query q được chuẩn hóa synonym. Nếu equivalent/hypernym-compatible thì không bác bỏ. Nếu sense không chắc thì UNKNOWN.

Trong v2, d chỉ là một competing description, không mang nhãn chân lý “negative”. Không cần tạo bảng disjoint tùy theo lỗi đã thấy trong cohort. Predicate “khác mô tả” không đủ để xóa box.

### 5.4 Region comparator và kiểm tra phụ thuộc ngữ cảnh

Ứng viên comparator: Alpha-CLIP frozen. Coder phải xác minh checkpoint/interface chính thức trước chạy; thiếu checkpoint hợp lệ thì báo BLOCKED, không âm thầm thay SigLIP.

RGB giữ pixel ảnh; alpha chỉ rõ region. Tránh black-out RGB quanh object vì tạo biến đổi ngoài phân phối không cần thiết. V2 dùng rectangle alpha từ candidate để có đầu vào nhất quán; actual SAM mask là ablation riêng, không gọi rectangle là mask vật thể.

Hai cách nhìn cố định:

- full: ảnh gốc và alpha của r;
- context: crop quanh r với hệ số 1.5, clip biên, ánh xạ alpha theo crop.

Hệ số 1.5 là lựa chọn thiết kế, không phải optimum đã chứng minh. Khóa trước run, không sweep theo kết quả.

Với mỗi view v, ghi cosine native:

m_v = similarity(v, observed_descriptor) - similarity(v, original_query).

Text template giống nhau cho hai phía. Không thêm thuộc tính cho query. Score summary:

z(r) = min(m_full, m_context).

Đây là margin bảo thủ qua hai views, không là confidence. Hai views có thể sai cùng nhau.

Quy tắc dự kiến: REMOVE khi descriptor hợp lệ, không compatible/unknown về sense, và z(r)>tau. Mọi trường hợp khác KEEP_UNVERIFIED. Baseline thực nghiệm đầu tau=0, strict >; đây là natural preference boundary, chưa được hiệu chuẩn an toàn. Không mặc định nó đã tốt.

### 5.5 Cycle grounding chỉ là ablation

Caption→ground→overlap dùng để đo binding, KHÔNG điều kiện semantic bắt buộc mặc định. Nó tăng chi phí và lặp chính objective CycleGRPO đã học.

Ablation +cycle có thể thêm IoU(g,r)>=0.5; ngưỡng hình học khai báo trước, không tương đương class accuracy. Grounding invalid → không remove trong ablation. Nếu không đem lại gain ở cùng target-retention thì bỏ cycle.

### 5.6 Đầu ra

Y = B trừ những box bị REMOVE. Count = |Y|. Box không bị sửa, không sinh ID instance mới cho một mask tưởng tượng. Lưu ledger cả box giữ và box bỏ; KEEP_UNVERIFIED không được diễn giải là đã chứng minh target.

## 6. Tác dụng và phần chưa giải

| Lỗi | Cơ chế v2 | Giới hạn |
|---|---|---|
| False-zero | A4 nguyên bản trước semantic gate | Có thể recovery nhầm query absent |
| Sai query/absent-positive | Observed descriptor + region comparator | Caption/comparator vẫn có thể sai chung |
| Box thiếu extent | Không sửa | Grounding ablation không tự làm representative |
| Duplicate/part–whole | Không sửa | Không tái dùng IoU/ownership rule đã fail |
| Undercount khi M0>0 | Không sửa | Cần đóng góp riêng, không giấu trong claim |

Nếu yêu cầu method phải giải cả năm ngay bây giờ thì v2 chưa đủ. Không có cơ sở thực nghiệm hiện tại để điền một identity resolver đáng tin. Thay vì lập trình một API rỗng gọi là solution toàn diện, bản này quy định đầy đủ đường triển khai và điều kiện để biết có đáng tiếp tục không.

## 7. Dataset và target mong muốn

### Dữ liệu chính

Ưu tiên FSC primary VAL và toàn bộ 111 verified-absent, đúng yêu cầu người dùng. Không tự mở LVIS mới. FSC100 cũ, FSC300 A4 và 111 đều đã được xem ở các mức khác nhau: ghi đúng development, không gọi confirmation mới.

Coder lập source-image overlap manifest cho mọi cohort. Các query cùng ảnh thuộc cùng split. Không cộng FSC100 và FSC300 nếu trùng ảnh. Baseline luôn đọc hoặc tái tạo đúng artifact tương ứng; không so 19.83 với14.95 như paired gain.

FSC point annotations đủ đo count/point localization, chưa đủ xác minh extent/owner. Các candidate không match point không tự thành semantic negative. Với 111 verified-absent, mọi prediction là false target theo protocol nhưng không biết nó là loại distractor nào.

Muốn semantic/identity claim cần một semantic audit có nhãn vùng độc lập: dùng artifact đã review nếu đủ; nếu chỉ assistant review, ghi exploratory. Không ép user annotation. MUCCA là external confirmation option sau khi khóa method, không tự tải/mở dùng để tune. Nếu chưa có nhãn chắc thì thu hẹp claim, không sinh pseudo-GT từ chính caption model.

### Mục tiêu và ngưỡng quyết định

Các mức dưới là mục tiêu thiết kế, KHÔNG forecast:

- Trên confirmation, primary MAE giảm ít nhất10% so M0; RMSE không tăng.
- Điểm TP của final set không thấp hơn M0; false-zero không tăng.
- So A4, cố gắng giữ >=99% maximum matching point TP của A4 trên development. Đây là non-inferiority budget kỹ thuật, không cam kết population safety.
- FSC111 absent mass giảm >=50%, activation giảm >=25% tương đối so M0. Với M0=111 activation, mục tiêu <=83; không gọi vậy là giải xong absence.
- Để claim giải absence mạnh, cần tốt hơn nữa trên verified-absent nguồn mới; không có con số nào trong v2 bảo đảm zero false positive.
- Query-level gains phải không chỉ đến từ một outlier: báo median, nhóm density và leave-one-source-out influence.

Quy tắc tiến tiếp: count/localization/absence phải cùng có lợi ở cohort đã khóa; không dùng giảm aggregate MAE do absent xóa mạnh để che primary mất TP.

## 8. Calibration hữu hạn, không thử vòng vô tận

Không huấn luyện; calibration threshold trên DEV là model selection, phải khai báo. Chạy feature extraction một lần, không chạy lại model để sweep prompt.

1. Báo tau=0 trước, cùng native score threshold control và Alpha-CLIP target-only control.
2. Nếu cần chọn tau, dùng source-split calibration với tối đa các quantile 0/25/50/75/90/95/100% của z hợp lệ và +infinity (không sửa), xác định từ features không đọc label trước.
3. Trên calibration, loại cấu hình giảm point TP >1% so base, tăng false-zero hoặc tăng primary MAE. Trong số còn lại chọn giảm absent mass nhiều nhất; hòa chọn tau lớn hơn. +infinity luôn là fallback.
4. Nếu chỉ +infinity đạt, trả NO_INCREMENTAL_VALUE, không đổi prompt/mask/model trong cùng version.
5. Threshold thắng phải freeze trước confirmation; selection trên labels DEV không được gọi zero-label calibration.

Chỉ đo margin phân biệt target–distractor ở các vùng có nhãn chắc. Nếu chỉ có count/point labels, kết luận end-to-end thay vì claim semantic accuracy.

## 9. Ablation bắt buộc để tách contribution khỏi model phụ

Trên cùng items:

- M0.
- A4 nguyên bản.
- M0 + Alpha-CLIP target-only filter, calibration cùng target-retention budget.
- M0 + observed-descriptor margin, full view only.
- M0 + two-view conservative margin.
- A4 + cùng two-view margin (main pipeline v2).
- +cycle grounding, chỉ sau khi comparator có tín hiệu để tránh trả phí cho cycle không cần.

Qwen/CycleGRPO YES/NO trực tiếp là control bổ sung nếu cache đã có; không trình bày thắng verifier yếu như bằng chứng thắng mọi verification.

Nếu full method không vượt target-only ở cùng TP-retention, không claim observed competitor là contribution. Nếu chỉ A4 có gain thì ghi gain thuộc recovery A4. Nếu chỉ thay model tăng điểm thì ghi gain thuộc pretrained model.

SAM3 B0+gate là extension riêng sau v2, cùng dataset/query. Không union A4+B0 cho đến khi có identity module được chứng minh; không sinh thêm module chỉ để bảng trông đầy đủ.

## 10. Implementation contract cho coder

Thư mục mới fqie_v2/; không sửa M0/A4 frozen artifacts.

Files cần viết:

- manifest.py: xác minh cache, hash, F1/F2 provenance, source overlaps, input/evaluation separation.
- base_predictions.py: M0 hoặc A4 nguyên bản; A4 trigger theo M0 trước gate.
- region_describe.py: adapter CycleGRPO native, deterministic generation, raw text cache.
- parse_descriptor.py: deterministic extraction/normalization; explicit UNKNOWN; unit fixtures.
- region_compare.py: Alpha-CLIP, full/context views, đúng image-alpha transforms, cosine scores.
- decide.py: chỉ quyết định từ saved evidence và locked tau; không đọc GT.
- evaluate.py: post-freeze count/point/absence và paired source bootstrap.
- calibrate.py: chỉ labels DEV được phép; xuất config riêng, không chạy trong inference.
- run.py: resume theo hash image/box/model/prompt/config; không reuse cache khác fingerprint.

Schema evidence: query_id, source_image_id, base_mode, candidate_id, F1_or_F2, box_original, region_kind, caption_raw_path, descriptor, parse_status, compatibility_status, scores_full, scores_context, z, tau, decision, reason, model_revision, transform_hash.

Không ép model trả JSON làm test capability. Parse free text theo native, lưu raw. Nếu parser không xử lý được noun đáng tin thì UNKNOWN, không retry bằng prompt đã sửa.

Khóa parser trước run semantic. Caption cache key không chứa query để tránh query leakage; proposal geometry vẫn phụ thuộc query nên không gọi evidence độc lập hoàn toàn với query.

## 11. Code phải đúng với ý tưởng

Kiểm tra trước run dài:

- M0/A4 parity: gate disabled → IDs/boxes/count exact bằng base.
- REMOVE-only → Y subset B và count không tăng; không thể tạo absent-positive mới so cùng base.
- A4 trigger không chạy lại sau pruning.
- Alpha RGB/alpha resize đồng bộ; synthetic rectangle/checkerboard kiểm coordinate mapping.
- Caption prompt không có query, labels, count, status. Log input literal để audit.
- Synonym/hypernym/unknown không thành contradiction; missing attribute không thành negative.
- Swapping q,d đổi dấu cosine margin (trên cùng embeddings); không đảo score nhầm.
- Strict z>tau; NaN/invalid/empty/truncated → KEEP_UNVERIFIED.
- All boxes processed; không âm thầm cắt topK theo query dày.
- Count đúng bằng độ dài final IDs; không tính count trước filter và box sau filter khác nhau.
- Freeze complete predictions trước label evaluation; missing item không tính0.
- Evaluator: maximum bipartite matching, một prediction tối đa một GT; đúng FSC radius/resize protocol đã khóa.

Tests này chứng minh implementation, không chứng minh semantic correctness.

## 12. Execution và báo cáo

Đầu tiên inventory cache rồi chạy smoke code/runtime trên ảnh dev có sẵn, gồm M0=0, M0>0, dense và absent. Không suy population từ smoke. Nếu runtime đúng, chạy toàn manifest development đã khóa: FSC primary và111, không chỉ2–4 ảnh.

Không đặt kill job 240/420 giây hay VRAM cap tùy tiện. Batch theo khả năng thực tế; resume, checkpoint response đều đặn; không tác động process của người khác. Theo quy ước bàn giao của bản họp, inference dài cung cấp lệnh cho user chạy; khi đã chạy được phép hoàn tất bình thường.

Báo thời gian thực đo, calls, số candidate, peak VRAM, parser unknown và coverage. Ước tính tổng chi phí theo số candidate thật và cả ảnh dày, không extrapolate từ một crop dễ.

Bảng mỗi run: cohort/version/n source/n query | M0 | A4 | proposed | ΔMAE | ΔRMSE | TP/FP/FN | false-zero | absent activation/mass | cost. Báo raw per-query CSV và source-bootstrap CI, không lẫn boxIoU50 với point metric.

## 13. Claim có thể đạt và claim không được nói

Nếu thành công: “Frozen region-conditioned verification improves the false-positive/recall trade-off of CountGD++ recovery on held-out sources, without counting-specific training.” Đây là claim điều kiện, cần số liệu vượt controls.

Không claim: learned region alignment mới; caption-ground cycle mới; training-free counting đầu tiên; full instance identity solved; chắc chắn A*; độc lập dữ liệu khi provenance pretrained chưa audit; no-target guarantee từ absence count thấp.

Zero-shot trong bài này: không user exemplar, inference image+text, không task-specific update. Không tự suy ra tất cả category/image chưa xuất hiện trong pretraining. Checkpoint training overlap chưa rõ phải ghi limitation.

Nếu v2 thành công nhưng identity chưa giải, publication claim phải hẹp đúng như trên. Nếu mục tiêu bắt buộc là novelty thuật toán lớn và giải toàn bộ identity, kết quả v2 chưa đủ. Không có lời hứa chắc chắn thay cho phần nghiên cứu còn thiếu.

## 14. Prompt giao coder

Đọc tài liệu này và bản họp03/11. Triển khai FQIE v2 bằng local frozen CountGD++/A4, CycleGRPO region description và Alpha-CLIP comparator, không training. Kiểm cache và source split trước; giữ M0/A4 nguyên bản. Hoàn thành manifest, adapter, two-view scores, deterministic decision, parity tests và post-freeze evaluator. Tách calibration khỏi inference. Không thêm ownership, graph, negative exemplar feedback hoặc model khác. Chạy smoke kỹ thuật rồi chuẩn bị run đầy đủ FSC primary development+111 absent với resume, cùng controls. Mỗi kết quả phải có bảng paired với M0 và A4. Nếu checker chỉ chọn no-op hoặc không vượt target-only ở cùng retention, ghi NO_INCREMENTAL_VALUE và không mở rộng/tune thêm trong version này. Bàn giao code/config/hash, predictions, evidence ledger, per-query CSV và quyết định theo target ở mục7.
