# AGENTS.md — Hướng dẫn Codex cho nghiên cứu CountGD++ zero-shot counting

> **Cập nhật hướng đang kiểm chứng (2026-10-20):** Kế hoạch hiện hành là
> [`target_aware_counting/output/method_validation_plan_20261020_v1/PLAN.md`](target_aware_counting/output/method_validation_plan_20261020_v1/PLAN.md).
> Các mục 3, 4 và 6 dưới đây giữ lại như lịch sử của hướng pseudo-exemplar gating;
> chúng không thay thế kế hoạch V0 full+tile, kiểm tra quan hệ instance và semantic
> safety đã được chốt. Thứ tự thực nghiệm hiện hành nằm trong `PLAN.md` nêu trên.
> P1 và P2 là câu hỏi độc lập; không bắt buộc cả hai phải thắng trước pilot count nhỏ.
> Mọi ràng buộc cứng ở mục 2 vẫn áp dụng.

## 1. Mục tiêu nghiên cứu đã chốt

Cải tiến CountGD++ thành bộ đếm **zero-shot, open-world, open-vocabulary, query-conditioned**: nhận một ảnh và một truy vấn văn bản bất kỳ, rồi đếm đúng các instance vật lý khớp với truy vấn đó. Truy vấn có thể chỉ một lớp đang hiện diện, một lớp đồng tồn tại với lớp khác dễ nhầm, hoặc một lớp vắng mặt. Mục tiêu cốt lõi là model phải đếm đúng *vật nào* theo query, không chỉ trả ra một con số hợp lý cho ảnh.

Giữ nguyên mục tiêu này trong mọi đề xuất và thay đổi. Prompt switching là phép thử để kiểm tra tính phụ thuộc vào query: giữ ảnh cố định, đổi query giữa các lớp hiện diện/vắng mặt, rồi kiểm tra điểm/box/instance được đếm có chuyển đúng theo query không. Nó là giao thức đánh giá, không yêu cầu hệ thống triển khai phải biết trước danh sách vật thể trong ảnh.

## 2. Ràng buộc cứng

- **Không fine-tune CountGD++ trên FSC-147.** Không huấn luyện head/verifier/module phụ trên nhãn FSC-147 trong roadmap hiện tại. Hướng đang theo là cải tiến inference-time; nếu muốn đổi sang bất kỳ hình thức học tham số nào, dừng và trình bày lý do trước.
- Giữ nguyên weights của CountGD++ trong các phép thử inference-time; ghi hash/đường dẫn checkpoint và xác nhận weights frozen.
- Không gọi LLM/VLM/API bên ngoài để tạo kết quả hoặc nhãn trong pipeline chính. Không thêm dịch vụ trả phí hoặc network dependency.
- Validation là tập phát triển. **Không chạy, xem kết quả, chọn threshold hay phát triển method trên TEST.** Chỉ mở TEST khi cấu hình và method đã khóa; ghi rõ đây là final evaluation.
- Không suy ra “absent” từ thiếu annotation. Mỗi cặp image–query phải có `present`, `verified_absent` hoặc `unknown`. `unknown` không bao giờ được tính là count 0 hay negative loss.
- Không tuyên bố cải thiện grounding/counting nếu chỉ có tổng count. Lưu và đánh giá localization/instance identity, false positive, duplicate/merged cases và query-switch behavior.
- Không gọi các pseudo boxes là ground truth. Gọi đúng là proposals/pseudo-exemplars; phân biệt kết quả tự sinh và nhãn GT được kiểm chứng.
- Không tuyên bố novelty, hiệu quả hoặc khả năng đạt conference A* trước khi có thí nghiệm đủ mạnh, baseline công bằng, ablation và đánh giá ngoài FSC-147.

## 3. Hướng method đang được kiểm chứng (chưa được xác nhận)

Giả thuyết hiện tại: CountGD++ có thể tự tạo pseudo-exemplars ở Forward-1 nhưng các proposal được đưa vào Forward-2 có thể sai lớp, trùng instance, hoặc bao nhiều instance. Thử một **inference-time, query-conditioned pseudo-exemplar validation/gating step** giữa hai forward: chỉ cho Forward-2 dùng proposal khi có bằng chứng ngữ nghĩa theo query, hỗ trợ visual cục bộ và tính toàn vẹn instance; proposal không đạt tiêu chí thì loại hoặc fallback rõ ràng. Backbone/model vẫn frozen; không học threshold trên TEST.

Đây là giả thuyết nghiên cứu, không phải method đã chứng minh. Không mặc định SAM là thành phần method; SAM2 từng dùng như công cụ chẩn đoán mask/quan hệ vùng. Chỉ đưa segmentation vào method nếu thí nghiệm chứng minh đóng góp riêng và chi phí hợp lý.

Các phép đo cần phân biệt:
1. **Localization/proposal recall:** GT instance có nằm trong proposals hay không và ở thứ hạng nào.
2. **Semantic alignment:** proposal có thực sự thuộc query class không, nhất là khi nhiều lớp cùng ảnh.
3. **Instance integrity:** proposal là một instance, proposal gộp nhiều instance, hay duplicate.
4. **Counting:** MAE/RMSE theo giao thức FSC; đồng thời báo coverage/abstention nếu gate từ chối dự đoán.
5. **Prompt switching:** trên cùng ảnh, đổi query giữa present co-occurring classes và verified-absent classes; lưu box/point được chọn, không chỉ count.

## 4. Roadmap thực nghiệm

Thực hiện tuần tự; mỗi bước có câu hỏi, giả thuyết, metric, artifact và điều kiện dừng.

### A. Khóa baseline VAL hợp lệ

- Xác minh đúng checkpoint, config, split, annotation JSON, image root, lệnh chạy và commit/diff của code.
- Xác nhận loader đọc `countgd_val_fscd147.json`/VAL đúng; không dựa vào tên folder hoặc tên run. Ghi số ảnh phải khớp split VAL (FSC-147 hiện dùng 1,286 VAL images theo các log đã thấy).
- Chạy baseline CountGD++ với pseudo-exemplar feedback tắt và lưu prediction theo từng ảnh: count, boxes/points, scores và metadata. Không ghi đè run cũ.
- So sánh baseline theo ảnh với các kết quả pseudo hiện có chỉ khi đã xác minh chúng cũng thực sự chạy trên VAL. Một audit dùng `instances_val.json` không tự chứng minh inference dùng VAL.

### B. Chẩn đoán cơ chế Forward-1 → Forward-2

- Đo proposal recall theo IoU và rank, score coverage, precision/purity, số proposal bị threshold/top-k, duplicate/merged/ambiguous và ảnh không có proposal.
- Các kết quả đã dán trong hội thoại là chẩn đoán thăm dò, chưa phải kết luận: ngưỡng thấp tăng coverage nhưng giảm purity; token score pooling chỉ tạo thay đổi nhỏ. Kiểm tra lại split, script và denominator trước khi dùng làm số liệu chính.
- Xem các lỗi cụ thể theo từng ảnh/query. Không chỉ hạ threshold để tăng coverage.

### C. Thử bộ lọc inference-time nhỏ, có ablation

Giữ baseline và tất cả weights cố định; so sánh cùng ảnh/query, cùng proposals và cùng evaluator. Bắt đầu với ablation tối thiểu:

- Native CountGD++ selection (baseline).
- Token-level query score trên proposal (max/mean/min chỉ là scoring ablations, không kết luận trước).
- Region-conditioned visual–text compatibility từ feature maps trước multimodal fusion, nếu adapter hiện tại lấy được đúng feature.
- Duplicate/overlap/merged-instance checks.
- Kết hợp từng tín hiệu rồi full gate; có fallback/no-feedback.

Threshold chọn trên VAL theo quy tắc đã định trước; báo kết quả paired per-image, bootstrap CI và tradeoff coverage–precision. Không tối ưu threshold theo từng lớp/ảnh test. Dùng cùng candidate pool để biết cải thiện đến từ ranking hay từ candidate generation.

### D. Đánh giá query switching có nhãn đúng

- Tạo bộ image–query có nguồn nhãn rõ: FSC primary class là present; các query khác chỉ absent sau xác minh độc lập; còn lại unknown.
- Trên cùng một ảnh, dùng prompt cho target chính, lớp khác đồng tồn tại (nếu đã kiểm chứng), và lớp absent đã kiểm chứng. Đánh giá localization và count cho từng query.
- Annotation tự động/VLM/SAM chỉ được dùng tạo gợi ý hoặc chẩn đoán, không tự nâng thành ground truth. Human review cần lưu nguồn nhãn và audit trail.
- Mở rộng từ pilot sang toàn bộ VAL và sau đó benchmark ngoài FSC-147 nếu dữ liệu/protocol cho phép. Pilot nhỏ chỉ để kiểm tra pipeline, không dùng làm bằng chứng hiệu quả.

### E. Khóa method rồi đánh giá TEST và dataset ngoài

- Sau khi lựa chọn xong bằng VAL, khóa code, config, threshold, metric và checkpoint; lưu commit/hash.
- Chỉ sau đó chạy TEST một lần để final report; không quay lại điều chỉnh method dựa trên TEST.
- Đánh giá dataset khác theo protocol zero-shot tương ứng, không huấn luyện/fine-tune trên dataset đánh giá. Nếu cần nhãn absent/co-occurrence thì xác lập riêng và không nhập nhằng với thiếu annotation.

## 5. Quy trình làm việc của Codex trong repo

### Binding dữ liệu FSC-147 cho artifact mới

- Từ 2026-10-14, mọi artifact/run **mới** dùng FSC-147 phải tham chiếu
  `/mlcv1/WorkingSpace/Personal/core_baotg/thu/xamxi/data/fscd147`.
  Config không ghi đè dùng cho run mới là
  `target_aware_counting/output/query_eval_contract_val_20261014_v1/data/datasets_fscd147_val_external.json`.
- Không sửa config snapshot/hash của run cũ chỉ để đổi path. Xác nhận hash GT
  `countgd_val_fscd147.json` vẫn là
  `6f9bc8d200d74789aa515e0131fd93275713392c64c3124828bdf258858ec807`.
- Default semantic rule vẫn là FSC category phụ `unknown`, không suy absent từ
  thiếu annotation. **Ngoại lệ protocol do người dùng khóa ngày 2026-10-14:**
  dataset dẫn xuất
  `target_aware_counting/output/query_eval_contract_val_20261014_v2_assumed_absent/`
  gán mọi category FSC không-primary là `assumed_absent`, GT=0. Không sửa
  FSC source và phải báo riêng `assumed_absent` với LVIS `verified_absent`.

Trước khi sửa hoặc chạy:

1. `pwd`, `git status --short`, `git rev-parse HEAD`; đọc `README`, config, entry point, data split. Không xóa/ghi đè thay đổi của người dùng.
2. Xác minh file tồn tại bằng `test -f`/`find`; không đoán path. Đọc help CLI và kiểm tra config thực tế.
3. Trình bày ngắn gọn câu hỏi thí nghiệm và output dự kiến trước lệnh chạy dài. Ưu tiên script có `set -euo pipefail`, output run mới có timestamp và không overwrite.
4. Đảm bảo split đúng bằng annotation path được log khi loader thật sự mở file, số ảnh và metadata trong config. Trước inference, in checkpoint path và kiểm tra tồn tại.
5. Sau lỗi, xử lý đúng lỗi đầu tiên; không chạy tiếp audit trên output thiếu/hỏng. Báo lỗi thực, không gọi một lệnh “pass” nếu chỉ compile/smoke test.
6. Với thay đổi code: backup/diff rõ; chạy unit/smoke test có ý nghĩa; xác nhận baseline output không đổi khi thêm instrumentation tùy chọn.
7. Sau thí nghiệm, lưu command, stdout/stderr, config snapshot, split, checkpoint, commit/diff, seed, run dir, số ảnh và metrics. Đọc lại summary trước khi diễn giải.
8. Nói rõ thí nghiệm đã hoàn tất hay mới dừng ở chuẩn bị/diagnostic; không yêu cầu người dùng lặp lại thông tin đã có. Người dùng thực thi GPU qua SSH khi môi trường Codex không truy cập được server; cung cấp block lệnh copy-paste gọn, tuần tự và có kiểm tra trước/sau.

## 6. Trạng thái hiện tại và việc tiếp theo

- Dự án chạy trên server người dùng tại `/mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC`; Codex trong phiên này không có SSH/GPU access mặc định. Không được giả vờ đã chạy trên server.
- Có patch trace Forward-1, script audit pseudo-exemplars và các probe scoring. `countgd_grounding/cgground/backend.py` đã được sửa tùy chọn `capture_token_scores`; smoke test trên ảnh `825.jpg` đã pass và cho thấy native boxes/scores không đổi khi bật capture.
- Một lần chạy full inference gần nhất thất bại trước inference do thiếu `CountGDPlusPlus/checkpoints/countgd_plusplus.pth`; vì vậy lần chạy đó không sinh kết quả. Có lần trước log 1,286 ảnh nhưng đường dataset khi đó từng bị nhầm VAL/TEST. **Chưa xác nhận bản pseudo run 20261008_210351 thực sự dùng VAL.** Trước mọi so sánh, audit `config_args_all.json`, lệnh và log annotation loader.
- Có lệnh VAL guarded đã được chuẩn bị: tạo `CountGDPlusPlus/config/datasets_fscd147_val.json` từ test config bằng thay annotation `countgd_test_fscd147.json` → `countgd_val_fscd147.json`; chọn checkpoint ở `checkpoints/countgd_plusplus.pth` hoặc `CountGDPlusPlus/checkpoints/countgd_plusplus.pth`; chạy no-pseudo baseline. Khi chạy, log bắt buộc phải nêu `countgd_val_fscd147.json`; nếu thấy test annotation thì dừng. Sau baseline, kiểm tra output và split trước khi audit.
- Việc ưu tiên: hoàn thành baseline no-pseudo trên VAL đúng split; sau đó tạo so sánh per-image với pseudo run chỉ khi pseudo run được xác minh VAL. Không chạy TEST để phát triển.

## 7. Quy tắc báo cáo

Mỗi báo cáo thí nghiệm phải có: (a) câu hỏi/giả thuyết, (b) split và cách xác minh, (c) checkpoint/code/config, (d) metric và denominator, (e) kết quả paired + độ bất định, (f) failure modes, (g) kết luận giới hạn trong phạm vi đã đo, (h) lệnh/artifact tái lập.

Dùng ngôn ngữ chính xác: “proposal recall tăng trên VAL subset” chứ không “grounding được giải quyết”; “zero-shot theo split/evaluation protocol của FSC-147” chứ không khẳng định khái niệm chưa từng xuất hiện trong pretraining. Báo rõ nếu kết quả là smoke test, exploratory diagnostic, subset hay full split.
