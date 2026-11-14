# Hai hướng method sau khảo sát nghiên cứu — 30/10/2026

**Trạng thái:** đề xuất nghiên cứu có cơ sở paper/source; chưa chạy checkpoint mới, chưa có gain của hai hướng này. Không cam kết novelty tuyệt đối, mức cải thiện hay acceptance A*. Đây không phải lệnh thay M0 hoặc thay đổi job RC đang chạy.

**Khuyến nghị:** triển khai hướng A trước. Hướng B là nhánh nghiên cứu riêng về vật nhỏ/dày, có thể dùng cùng model nhưng không tự động ghép vào A. Count Anything là đối chứng mới đáng bổ sung, không phải đóng góp của nhóm.

## 1. Bài toán và điều đã biết

Input là ảnh + text query. Output phải là một tập instance có vị trí kiểm tra được; count bằng cardinality của chính tập đó; target không có phải trả tập rỗng. Nếu claim box-level grounding thì phải có một box hợp lệ cho mỗi instance và đánh giá extent, không thay IoU50 bằng point-in-box.

Giữ ràng buộc không học thêm tham số. Được sử dụng checkpoint pretrained frozen; phải ghi đúng dữ liệu huấn luyện đã công bố và phần overlap chưa xác minh. Không tự coi frozen là chưa từng thấy category/dataset. Không dùng GT/status lúc inference.

Các số dưới đây lấy từ báo cáo người dùng cung cấp, không phải kết quả vừa tái chạy hoặc đọc trực tiếp trên máy /mlcv1:

| Cohort development đã có | M0 MAE | Localization | Absence / false zero |
|---|---:|---|---|
| LVIS 300 query / 100 ảnh, 200 present + 100 verified absent | 1.180 | TP50 154/409, FP183, FN255 | absent-positive 3/100; present false-zero 88/200 |
| FSC primary VAL 100 ảnh/query | 19.83 | point TP4077, FP478, FN2013 | false-zero 14/100 |
| FSC verified absent 111 pair | 62.54 | 6942 box dự đoán | absent-positive 111/111 |

Đây chưa phải baseline toàn bộ FSC147 hoặc toàn bộ LVIS. RMSE phải lấy từ artifact đúng cohort, không tự điền số của run khác. FSC primary, 111 verified negatives, và FSC auxiliary assumed-absent là các protocol khác nhau.

Raw oracle LVIS đạt 366/409 GT; đó là dư địa có GT chọn, không phải bằng chứng đã có scorer đủ tốt. F1 raw và F2 M0 là hai forward khác nhau: giữ riêng provenance và ID.

Các thất bại cũ định hướng thiết kế mới:

- Score/cosine và vote không cung cấp thêm bằng chứng đúng target/đúng instance một cách đáng tin.
- Tăng coverage hoặc cộng tile prediction không đồng nghĩa cải thiện count.
- Xóa nhiều prediction có thể giảm absent MAE nhưng làm mất target thật; mean-24 sink đã cho thấy điều này.
- Qwen chọn action qua option/permutation và CycleGRPO ownership chưa chứng minh được quyết định tập instance.
- RC hiện có vấn đề hội tụ và target-only gain. Giữ kết luận về run đó theo artifact cuối; không ngắt hoặc sửa job người dùng đã quyết định cho chạy hết.

## 2. Những nghiên cứu mới thay đổi lựa chọn

| Nghiên cứu / nguồn chính | Cơ chế liên quan | Hệ quả cho dự án |
|---|---|---|
| MolmoPoint, 03/2026 [1–3] | Native grounding tokens, dự đoán vị trí theo chuỗi và lớp no-more-points | Có head đã học chọn vị trí và dừng; đáng khảo sát hơn tiếp tục hỏi VLM YES/NO. Không phải chứng minh accuracy trên FSC/LVIS của mình. |
| Count Anything, 05/2026 [4–5] | Counter cho vùng thưa và counter điểm cho vùng dày, output tập điểm | Baseline chuyên dụng mới; CLOC có nguồn FSC147/FSCD-LVIS nên phải kiểm tra split. Không gọi việc ghép hai nhánh là novelty mới. |
| CountEx, ECCV2026 [6] | Học tách phần feature chung và phần đặc trưng của distractor để exclusion | Semantic exclusion có tiền lệ mạnh; lấy module này rồi ghép vào CountGD không tự thành đóng góp mới. |
| AdaCount, 07/2026 [7] | Prototype similarity hướng spatial warping và feature modulation cho SAM3 | Đã có training-free spatial adaptation. Hướng B phải phân biệt bằng cơ chế, không đổi tên warping. Trang tác giả còn ghi code coming soon khi khảo sát. |
| From Seeing to Reporting, 10/2026 [8] | Phân biệt response, records, sets, readout; candidate coverage khác final selection | Count từ tập vị trí là yêu cầu hợp lý nhưng đã có tiền lệ. Kết quả synthetic/API không chuyển nguyên thành gain FSC/LVIS. |
| Why VLMs Miss Small Objects, 10/2026 [9] | Tách chi tiết trên mỗi vật khỏi lượng nội dung mỗi call phải xử lý | Nhiều pixel/zoom không bảo đảm mọi metric tốt hơn; không dùng lý thuyết recall có điều kiện để claim absence safety. |
| Minimum Bayes risk decoding và GOSPA [10–11] | Chọn output theo expected set error, tính cả localization/missing/false instances | Có nền tảng cho quyết định tập ở A; hai cơ chế này không phải phát minh của nhóm. |

Khảo sát này kiểm tra paper, model card, repo và một đường source quan trọng; không phải chứng nhận đã khảo sát mọi công trình, chưa có reproduction runtime. Các nguồn rất mới là preprint nếu chưa có venue được xác minh.

## 3. Hướng A — Quyết định tập instance từ phân phối grounding trực tiếp

### 3.1 Giả thuyết

Thay vì hỏi từng box có đúng hay không, đánh giá các **tập vị trí** có thể là câu trả lời, gồm cả tập rỗng. Điểm quyết định đến từ một model đã được học để chọn vị trí theo query và theo các vị trí trước đó. CountGD++ cung cấp các nhánh tìm kiếm tiềm năng; không quyết định số lượng bằng threshold của từng candidate.

Ví dụ, với ba proposal trên hai chiếc cốc, các giả thuyết cần cạnh tranh là hai vị trí cốc, ba vị trí, một vị trí, hoặc không có vị trí. Không coi ba box đều có hình cốc là bằng chứng có ba vật.

**Model:** CountGD++ frozen + `allenai/MolmoPoint-8B` frozen, bản ảnh tổng quát. Không dùng bản GUI/Video thay thế tùy tiện. Đây là thay đổi module ra quyết định; không lấy tổng hoặc trung bình count của hai model.

### 3.2 Chi tiết source đã xác minh

Source HF `modeling_molmo_point.py` trả `patch_logits`, `subpatch_logits`, `location_logits`. Nhưng wrapper generation chỉ giữ patch argmax trong đường logits mở rộng. Vì thế `generate(num_beams=...)` đơn thuần không thực hiện beam search đầy đủ trên patch. Cần adapter đọc các head riêng và cache trạng thái theo nhánh. Đây là quan sát source, chưa phải runtime parity [3].

### 3.3 Pipeline

1. **Đọc proposal có provenance.** Dùng cache F1 đã có; M0 F2 chỉ là control/nguồn giả thuyết tham chiếu. Không bỏ mọi raw score thấp bằng ngưỡng 0.23 trước nghiên cứu selection. Mọi giới hạn candidate phải nằm trong config và có audit coverage, không đổi theo GT.
2. **Encode ảnh sạch và query bằng MolmoPoint native.** Một prompt cùng nghĩa, cố định cho mọi ảnh của query. Không vẽ option A/B/C, không đưa GT count, không yêu cầu model tự viết số đếm. Giữ metadata processor để quy về tọa độ nguồn.
3. **Tạo các chuỗi grounding thay thế.** Mỗi prefix mở rộng bằng các nhánh native có điểm cao, thêm các nhánh ở vùng CountGD có thể bị native greedy bỏ qua, và nhánh dừng. Không ép chọn điểm bên trong mọi box; không ép số điểm tối thiểu. Native proposals ngoài CountGD vẫn được phép, để pool CountGD không trở thành trần cứng.
4. **Chấm toàn bộ chuỗi bằng phân phối thực.** Cần tích lũy log-prob của quyết định ngôn ngữ/cấu trúc nếu có, patch, subpatch, location và dừng. Grammar và thứ tự theo native; dùng đúng masks native. CountGD chỉ điều khiển nơi tìm kiếm, không cộng bonus vào xác suất.
5. **Chọn một tập cuối.** Control chọn chuỗi hoàn chỉnh có xác suất cao nhất. Nhánh method chọn tập có expected set error nhỏ nhất trong ngân hàng giả thuyết. Giữ tập rỗng với điểm thực; không cho nó điểm 0 mặc định và không loại nó khỏi cạnh tranh.
6. **Xuất vị trí, count, trace.** Mỗi điểm có nguồn chuỗi/token và các candidate liên quan. Count là số điểm cuối. Không thêm count correction sau khi đã xuất vị trí.

Các nhánh dùng chung image cache khi implementation cho phép; trạng thái autoregressive và last-point state phải riêng từng nhánh. Trước tiên dùng exact implementation đơn giản; tối ưu cache sau khi parity đã đạt.

### 3.4 Quyết định tập: cơ sở và công thức

Gọi H là ngân hàng chuỗi hoàn chỉnh, mỗi chuỗi giải mã thành tập điểm S. Điểm log-likelihood phải gồm cả dừng. Những chuỗi trùng hoàn toàn không được nhân trọng số vì bị đưa vào nhiều lần từ proposal khác nhau.

Với trọng số chuẩn hóa trong ngân hàng `w(T)`, chọn:

`S* = argmin_{S in H} sum_{T in H} w(T) D(S,T)`.

`D` là GOSPA trên tọa độ ảnh nguồn: matching một-một, chi phí khoảng cách có giới hạn, cộng phạt điểm thừa/thiếu. Tập rỗng hợp lệ. Đây là áp dụng minimum Bayes risk, không phải yêu cầu ba lần trả lời giống nhau [10–11].

**Giới hạn toán học phải ghi rõ:** `w` là phân phối xấp xỉ bị giới hạn bởi H, không phải posterior đã hiệu chuẩn trên thế giới thật. Giảm expected loss theo model không chứng minh giảm lỗi GT. Cutoff khoảng cách của GOSPA là hyperparameter thực, phải khai báo và kiểm tra sensitivity; nó không chứng nhận hai điểm là cùng instance. MAP có length bias; MBR có thể chọn thiếu nếu hypothesis bank bị lệch. Không thêm temperature/length bonus để ép count đẹp sau khi xem kết quả.

Không lấy softmax chỉ trên các vùng CountGD còn lại: việc đó có thể biến một đám candidate sai thành xác suất dương cao. Native normalization và nhánh no-more-points phải được giữ. Nếu có importance sampling thay vì beam, cần trọng số đúng proposal distribution; không dùng score tùy ý làm posterior.

### 3.5 Nó đóng góp vào problem nào?

| Problem | Cơ chế thử nghiệm | Điều chưa được bảo đảm |
|---|---|---|
| Sai query / absent | Query-conditioned native spatial choice cạnh tranh với dừng | Model vẫn có thể hallucinate đúng kiểu distractor |
| Box trùng / part | Quyết định điểm tiếp theo có điều kiện trên tập đã mô tả; set loss phạt điểm dư | Native chống lặp token không phải physical identity; hai phần có thể ở hai token |
| Missed low-score candidate | CountGD mở nhánh tìm kiếm nhưng không là semantic veto | Không cứu vật model không nhận biết hoặc không nhìn rõ |
| Count và vị trí bất nhất | Count lấy trực tiếp từ tập điểm được chọn | Không tự chứng minh box extent |

### 3.6 Box extent: không được lách mục tiêu

MolmoPoint trả point, không phải whole-object box. Nếu protocol cần box, A **chưa là solution hoàn chỉnh chỉ nhờ point tốt**.

Giữ hai kết quả tách bạch: count+point và count+box. Một renderer đối chứng có thể gắn điểm với proposal CountGD theo assignment một-một có unmatched; các lựa chọn extent cạnh tranh phải được lưu. Point nằm trong box không chứng minh box ôm đúng toàn vật.

Có thể dùng SAM2 frozen làm renderer có positive point và negative points của các instance lân cận, với cùng điểm đã chốt, để thử cải thiện extent. Đây chỉ là ablation extent đã có tiền lệ; không dùng mask để sinh thêm/bớt instance hoặc quyết định count. Nếu point sai/part, SAM không được coi là người sửa semantic tự động. Không có box đáng tin phải ghi unmatched/extent failure, không bỏ query khỏi denominator và không tuyên bố đạt contract box-level.

### 3.7 Novelty có thể là gì?

Phần đang đề xuất là: **bảo toàn phân phối spatial choices trước khi greedy collapse, tìm kiếm có dẫn đường bởi raw proposals, rồi ra quyết định dưới loss của tập instance với empty hypothesis**.

MolmoPoint, beam search, MBR, GOSPA, point prompting và proposal generation đều đã có. Chưa tìm thấy một bản trùng nguyên cấu hình này trong khảo sát, nhưng điều đó không đủ chứng nhận mới. Claim phải chứng minh giá trị vượt native MolmoPoint, đúng native beam cùng compute và simple two-model filtering. Nếu chỉ hơn M0 nhờ đổi model thì đó là baseline mới, không phải method mới của nhóm.

## 4. Hướng B — Phân bổ độ phân giải theo giới hạn phân biệt vị trí

### 4.1 Giả thuyết độc lập

Đối tượng thật nhỏ/sát nhau có thể rơi vào cùng đơn vị spatial decoding. Khi đó ngưỡng semantic tốt hơn vẫn không đủ để xuất hai vị trí riêng. Cần thay đổi phần ảnh được cấp độ phân giải, thay vì nới score hoặc cộng kết quả tile.

Native MolmoPoint cấm hai điểm dùng cùng IMAGE token và ViT subpatch [1]. Đây là giới hạn của **biểu diễn token**, không phải quy luật mỗi vật chỉ chiếm một patch. Vì processor có thể tạo nhiều crop, phải đo toàn bộ mapping thật; không suy ra ngay “mỗi ô 14×14 pixel ảnh gốc chỉ đếm được một vật”.

### 4.2 Pipeline

1. Chạy processor native để lấy mapping pixel nguồn → các patch/subpatch có thể chọn.
2. Chiếu các vị trí đề cử CountGD và support native lên mapping này. Xác định vùng có nhiều vị trí đề cử phân biệt trong ảnh nguồn nhưng ít địa chỉ decoder phân biệt. Chỉ gọi đây là **nhu cầu độ phân giải ước lượng**, không gọi là số vật thật.
3. Xây ánh xạ không gian thuận nghịch, liên tục, một-một; tăng diện tích biểu diễn ở vùng có collision, giữ một phần ngân sách đồng đều toàn ảnh để không xóa nơi CountGD bỏ sót.
4. Render lại **một ảnh** từ pixel nguồn bằng ánh xạ đó, dùng model frozen để decode một tập điểm. Không cộng count ảnh gốc và ảnh warp.
5. Đổi tọa độ điểm về ảnh gốc bằng inverse map, rồi đánh giá count và localization như thường.

Bản đầu có thể dùng hai ánh xạ đơn điệu riêng theo trục x/y, xây từ marginal của demand map với uniform floor; derivative có cận để tránh ép một vùng về gần zero. Xác định cận trước run, ghi số thực tế, không giả vờ đây là tham số không cần chọn. Uniform demand phải trả identity transform. Đây là thuật toán hình học, không tối ưu weights bằng nhãn.

Nếu marginals làm méo quá mạnh hoặc demand trải đều khiến transform gần identity, không kỳ vọng gain. Không xóa native repeated-token protection chỉ để ép model xuất nhiều điểm.

### 4.3 Vì sao khác các phép resize/tile đã thử?

Uniform resize tăng tài nguyên ở mọi nơi; union tile sinh nhiều tập prediction phải hòa giải. B thử giữ một tập đầu ra và cấp độ phân giải theo **độ phân giải địa chỉ của decoder**. Tuy nhiên AdaCount đã dùng spatial warping cho counting [7]; “warp ảnh” không phải novelty. Điểm khác có thể kiểm chứng là allocation dựa trên collision/capacity của output representation, thay vì prototype semantic similarity.

### 4.4 Rủi ro và phạm vi

- Nhiều proposal có thể chỉ là duplicate/part, nên demand cao có thể sai. Warp không xác minh target.
- Warp thuận nghịch giữ topology nhưng có thể làm biến dạng appearance và đổi phân phối input pretrained.
- Ở budget cố định, ưu tiên một vùng làm vùng khác ít chi tiết hơn.
- Processor nội bộ có thể resize/crop làm mất lợi ích; đo effective mapping sau processor, không chỉ nhìn ảnh warp.
- Model có thể đếm nhiều distractor hơn khi chúng rõ hơn.
- Nếu lỗi dominant là semantic hoặc stopping, B không phải cách sửa trực tiếp.

**Ưu tiên thấp hơn A.** Chỉ đưa B thành thành phần chính khi có gain count+point trên FSC dày ở cùng input budget, đồng thời không làm hỏng LVIS/111 negatives. Không dùng B để che một verifier semantic yếu.

## 5. Đối chứng mới đáng chạy nhưng không gọi là phát minh

Count Anything [4–5] có checkpoint công bố và nhiệm vụ text→point set gần mục tiêu. Nên bổ sung standalone để biết nền tảng hiện hành đã tiến tới đâu. Không ghép output nó với CountGD trước khi có lý do.

CLOC dùng nguồn FSC147 và FSCD-LVIS. Cần đối chiếu source-image/split của checkpoint với evaluation manifest; cùng nguồn không tự chứng minh leakage, nhưng cũng không thể tự gọi toàn bộ ảnh của mình là unseen. Nếu overlap chưa xác minh, gắn nhãn rõ khi báo số. Không lấy số CLOC từ paper đặt cạnh M0 LVIS300 như cùng protocol.

CountEx và HieraCount là các nguồn liên quan semantic granularity, không phải license để biến module đã học của họ thành contribution riêng. AdaCount là nearest prior của B, phải so nghiêm túc khi code khả dụng; không tự viết adapter rồi gọi đã reproduce paper.

## 6. Coder cần triển khai gì

### A. Runtime và correctness có mục tiêu

- Dùng checkpoint MolmoPoint-8B chính thức, khóa revision và processor. Không thay bằng Qwen3-VL hoặc checkpoint GUI nếu thiếu dependency.
- Adapter native greedy và evaluator point/count; lưu completion/truncation/error khác empty. Output-token limit phải đủ cho dữ liệu dày và nằm trong context model; không sao chép `max_new_tokens=200` của demo rồi báo là count đầy đủ.
- Hook ba grounding heads trước argmax wrapper. Build branch state gồm language KV, image cache, last selected patch, repeat mask, coordinate metadata.
- Kiểm tra chế độ width=1 theo đúng **native greedy policy** khôi phục native output; phân biệt nó với joint maximum-probability decoding qua cả ba head, vốn có thể cho lựa chọn khác. Kiểm tra step log-prob bằng tính trực tiếp; không chỉ kiểm parser.
- Kiểm tra box/source → token → source; crop/padding/warp inverse; `DONE` sau 0 và sau N điểm; empty không bằng malformed; count bằng số điểm đã xuất.
- Với cache chưa đúng, reference implementation branch tuần tự có thể dùng trước. Không giả định `num_beams` HF tự quản lý đúng custom image state.

### B. Các nhánh bắt buộc so công bằng

| Nhánh | Mục đích |
|---|---|
| M0 hiện có trên cùng items | So baseline dự án |
| MolmoPoint native greedy | Tách lợi ích đổi model |
| MolmoPoint proper spatial beam, cùng compute | Tách lợi ích search thông thường |
| Proposal-guided search + MAP | Tách giá trị đề cử CountGD |
| Proposal-guided search + set-risk decision | Kiểm đóng góp ra quyết định tập A |
| Native + uniform resolution tăng tương đương budget B | Đối chứng chi phí/chi tiết ảnh |
| Native + capacity allocation | Kiểm B độc lập |
| A+B | Chỉ cần khi từng thành phần đáng giữ; không mặc định là method cuối |

Ngân hàng giả thuyết và loss là algorithmic config, không phải giới hạn chạy 240/420 giây. Cho job chạy bình thường, resume, log cost; không dừng/chiếm GPU của người khác. Không có MILP hoặc bài toán all-subsets. Matching của set loss dùng thư viện assignment, cần báo chi phí thật ở các ảnh dày; không dự đoán runtime cả bộ từ một ảnh dễ.

### C. Dataset và báo cáo

Chạy các cohort development đã có thay vì tiếp tục tạo chuỗi cohort 2–4 ảnh để tuyên bố capability:

1. LVIS300 có present và verified-absent.
2. FSC primary VAL100.
3. Toàn bộ 111 verified negative pairs.

Smoke nhỏ chỉ để phát hiện lỗi implementation. Sau đó chạy paired evaluation đầy đủ các cohort trên với config khóa; mọi item nằm trong denominator, cả M0=0. Không chọn ảnh bằng gain. User đã có baseline thì dùng cache đúng hashes, không yêu cầu tái lập vô cớ.

Khi một cấu hình có lợi ích end-to-end, khóa cấu hình rồi mở rộng **full FSC VAL và LVIS VAL theo query protocol đầy đủ đã công bố**, tiếp đến final held-out TEST theo lịch khóa. “Nguyên bộ LVIS” không có nghĩa mọi category không được annotation đều là absent. Manifest phải xác định present exhaustive và verified negatives. Không gọi 111 pair là benchmark độc lập mới.

Mỗi bảng luôn có M0 và delta trên cùng items: MAE↓, RMSE↓, exact, TP/FP/FN, present false-zero, absent positive total/new/repaired/count mass, query-switch, runtime và input/output token. FSC point metric và LVIS TP50 không trộn. Với output điểm mới, thêm metric point chung cho M0/model để so công bằng nhưng giữ TP50 cho claim box. Phân tích low/high density và khoảng cách instance, bootstrap theo source image.

### D. Kết quả nào mới đáng tiếp tục?

- A cần vượt M0 **và** native MolmoPoint/proper beam ở count cùng localization, đồng thời không tái diễn đổi recall lấy absence. No-more-points là khả năng đã học, không phải safety guarantee.
- B cần tốt hơn uniform resolution/tiling cùng budget và giữ absence; chỉ giảm collision proxy chưa đủ.
- Nếu Count Anything standalone thắng nhưng A/B không thắng model nền tương ứng, báo đúng là cập nhật strong baseline; không viết claim inference innovation.
- Nếu điểm tốt nhưng box TP50 xấu, chỉ có tiến bộ count+point, chưa giải đủ contract count+box. Cần giữ kết luận này, không đổi metric để qua cổng.

## 7. Mức tin cậy và quyết định

| Nội dung | Mức chứng cứ hiện tại |
|---|---|
| MolmoPoint có head grounding và dừng; source có argmax wrapper | Đã kiểm paper/model card/source công khai |
| Có thể viết inference search từ các head này | Có đường source khả thi; runtime/cache parity chưa kiểm |
| Set-risk decision cải thiện FSC/LVIS | Giả thuyết chưa có kết quả |
| Capacity allocation tốt hơn AdaCount/uniform detail | Giả thuyết rủi ro cao hơn; nearest prior đã biết |
| Hai hướng bảo đảm A* hoặc novelty tuyệt đối | Không có căn cứ để cam kết |

Chọn **A là câu hỏi method chính**: liệu giữ và khai thác phân phối grounding theo tập có giải quyết đồng thời chọn đúng target, cardinality và absence tốt hơn greedy/filter không? B là nhánh bổ trợ riêng về năng lực phân giải ảnh. Không mở thêm Qwen-router, SigLIP-margin, DINO-matching, sink hoặc prompt sweep từ tài liệu này.

## Nguồn kiểm tra

1. MolmoPoint, paper: https://arxiv.org/html/2603.28069v1
2. MolmoPoint-8B model card: https://huggingface.co/allenai/MolmoPoint-8B
3. Native HF source: https://huggingface.co/allenai/MolmoPoint-8B/blob/main/modeling_molmo_point.py ; official repo: https://github.com/allenai/molmo2/blob/main/MOLMO_POINT_README.md
4. Count Anything, paper: https://arxiv.org/html/2605.30846v1
5. Count Anything source/model/data: https://github.com/Mengqi-Lei/count-anything ; https://huggingface.co/MengqiLei/count-anything ; https://github.com/Mengqi-Lei/count-anything/blob/main/data/README.md
6. CountEx official: https://github.com/bbvisual/CountEx ; paper: https://arxiv.org/abs/2602.19432
7. AdaCount author project: https://muhammad-ibraheem-siddiqui.github.io/AdaCount/ ; paper: https://arxiv.org/abs/2607.02139
8. From Seeing to Reporting, 10/2026: https://arxiv.org/abs/2610.25503 ; official code: https://github.com/shijunzhe/vlm-set-reporting (abstract/repo được kiểm; toàn bộ PDF chưa đọc trong khảo sát này).
9. Why VLMs Miss Small Objects, author research page: https://quotr.ai/research/small-object-vision/ ; paper: https://arxiv.org/abs/2610.09313 (nguồn tác giả/abstract; không coi đã reproduce định lý trên CountGD).
10. Minimum Bayes risk: https://proceedings.iclr.cc/paper_files/paper/2025/hash/1588dc2b2ef339d6e4c47d212e36f991-Abstract-Conference.html ; https://aclanthology.org/2025.acl-long.793/
11. Generalized optimal sub-pattern assignment metric: https://arxiv.org/abs/1601.05585

