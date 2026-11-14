# Hướng dẫn chạy

## 1. Chạy lại kết quả bằng CPU
Yêu cầu Python 3.12, numpy và pandas.

    pip install -r requirements.txt

    python scripts/p17_01_three_view_paired_eval_v1.py --extra-csv outputs/joint_semantic_counting_v1/p14_08b_09_joint_unified_frozen_v1/unified_per_pair_predictions.csv

Kết quả nằm trong `outputs/joint_semantic_counting_v1/p17_01_three_view_paired_v1/`.

## 2. Chuẩn bị FSC147 + 111 absent
Tải FSC147 từ nguồn chính thức, giải nén vào `data/fscd147/FSC147_384_V2/`.
File `test_pairs_1301.csv` đã có sẵn 1190 positive và 111 absent pairs. File `annotation_absent111_empty.json` chứa annotation absent. Không cần tải thêm 111 ảnh: 111 pairs sử dụng 96 ảnh có trong FSC147.
Giữ nguyên `pair_id`, `file_name`, `query`, `count_target` và các image paths trong manifest.

## 3. Chạy inference GPU
CountGD++ upstream: https://github.com/niki-amini-naieni/CountGDPlusPlus (commit `8489f83`).
Checkpoint và intermediate graph/decoder caches không nằm trong repo. Full inference trên máy sạch **chưa được xác minh**; không dùng bước CPU ở trên để khẳng định đã tái lập inference.
