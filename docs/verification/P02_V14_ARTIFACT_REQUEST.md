# Yêu cầu artifacts gửi Thúy / P01 để mở khóa P02 parity

P02 đã chạy GPU smoke thật với query-only, `num_exemplars=0`, pseudo=3.
**Đây chưa phải V14 parity.** Nhờ Thúy cung cấp các tài sản nguyên bản dưới đây;
không cần GT để chạy adapter và không mở `FINAL_SEALED` cho lượt smoke này.

1. **Raw V14 gốc và checksum:** `full_cohort_output.json` đúng run đang được
   `FQIE_BIDIRECTIONAL_HOLDOUT_RESULTS_20261116.md` tham chiếu. SHA256 đang ghi là
   `94e71b3f9668e122aefa698480189081284fe0095ee3a0b6c25bfe1b85a24fd9`.
   Cần file thật, không chỉ summary. Nếu hash khác, gửi nguồn/thời điểm/lý do,
   không ghi đè artifact cũ hoặc sửa hash để hợp thức hóa. Nếu file chứa GT,
   giữ nguyên bản evaluation riêng và cung cấp export inference GT-free bên cạnh.
2. **Baseline predictions GT-free theo từng query:** `image_id`, `query_id`,
   query nguyên văn, `baseline_count` dự đoán, raw detector-query indices,
   boxes trước/sau filtering, scores, size ảnh gốc/resize và hệ tọa độ.
   Ghi rõ score pooling, token mask, clipping/NMS, cropping/TT-Norm và stage
   được dùng để tính baseline. Nếu raw logits, first/second-pass output hoặc
   indices chưa từng được lưu, ghi rõ thiếu; không tái dựng boxes từ count/density.
3. **Source lock:** URL upstream, commit đầy đủ, toàn bộ diff/custom patches,
   entrypoint/lệnh đã chạy và config snapshot. Bao gồm code chọn pseudo,
   loader/preprocessing, caption/tokenizer. Xác nhận số exemplar ban đầu,
   số pseudo, synthetic/negative exemplars, số forward và mọi adaptive branches.
   Không mặc định source P02 `2ecbc433...` chính là source V14.
4. **Weights/BERT lock:** checkpoint SHA256 + kích thước + provenance
   (tác giả hay mirror, đã kiểm tra equivalence chưa); BERT directory inventory
   với SHA256 của config, vocab, tokenizer files và BERT weights. Chỉ cung cấp
   đường dẫn local/kênh trao đổi phù hợp, không commit weights hoặc credentials.
   Checkpoint PrACo `b89395d7...` hiện có không tự chứng minh đúng checkpoint V14.
5. **Runtime/build lock:** Python, torch/torchvision/transformers/timm/NumPy/Pillow,
   CUDA/cuDNN/NVCC/GCC, device, precision/AMP, seed, TF32, deterministic/benchmark
   flags; binary SHA của deformable-attention op, source manifest + patch,
   build command/flags. Gửi timestamps UTC thật của run/freeze và thời điểm mở
   cohort; tên thư mục hoặc tên file có ngày không thay thế provenance.
6. **Cohort nhỏ được phép dùng:** 20–50 image/query pairs GT-free, ảnh gốc hoặc
   đường dẫn truy cập được, SHA256 từng ảnh, split/source-image mapping và
   checksum toàn bộ roster. Cần reference predictions đầy đủ cho cùng roster.
   Không chọn mẫu dựa trên kết quả FQIE hay chỉ chọn ca đã thắng; không đưa
   present/absent/count labels vào manifest inference; tránh `FINAL_SEALED`.
7. **Duyệt phép đối chiếu trước khi mở raw:** count phải khớp từng query;
   box matching theo native query index, tolerance đề xuất `atol=1e-4` pixel gốc,
   `rtol=0`; score `atol=1e-6`, `rtol=0`; coverage 100%, không bỏ missing/extra
   candidates. Kế hoạch ở `configs/integrations/official_pseudo3_parity_plan.json`
   còn là đề xuất, cần review trước; không nới tolerance sau khi thấy mismatch.

GT/count/presence labels phục vụ evaluation phải ở file riêng, chỉ đọc sau khi
predictions được freeze. Nếu source/config V14 khác operating point P02 hiện tại,
hai bên cần review implementation được version riêng; không đổi tên PrACo FULL
hoặc native NO-PSEUDO thành V14, không sửa giao thức/artifacts frozen.
