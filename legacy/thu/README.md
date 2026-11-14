# Joint Semantic Counting

Nghiên cứu cải thiện CountGD++ bằng Joint V3.2 (HOW MANY) và A3 (WHAT).

Unified 1301 pairs: CountGD++ MAE **15.9908**, Joint V3.2 + A3 MAE **13.2280**.

## Các file

- `scripts/audit_decoder_query_embeddings.py`: Kiểm tra và trích xuất decoder features.
- `scripts/experiment_joint_v3_p11_reliability.py`: Module phụ thuộc của pipeline.
- `scripts/p14_07b_frozen_a3_fsc_unified.py`: Semantic gating bằng A3.
- `scripts/p14_08a_extract_frozen_v4_nodes.py`: Trích xuất contribution của V4.
- `scripts/p14_08b_09_joint_unified.py`: Tái hiện logic Joint V3.2 từ các cache đã đóng băng.
- `scripts/p17_01_three_view_paired_eval_v1.py`: Tính MAE, RMSE, Exact Accuracy và HELP/HARM.
- `scripts/train_instance_relation_encoder_v1.py`: Module phụ thuộc của pipeline.
- `scripts/train_instance_relation_encoder_v3.py`: Module phụ thuộc của pipeline.
- `scripts/train_instance_relation_encoder_v4.py`: Module phụ thuộc của pipeline.
- `scripts/train_s1_semantic_verifier_a3_clip_v1.py`: Định nghĩa và huấn luyện semantic verifier A3.

- `generated/fsc147_presence_v4/`: manifest và nhãn 111 absent.
- `outputs/`: frozen predictions và feature cache, không có checkpoint.
- `RESEARCH.md`: giải thích nghiên cứu.
- `RUN.md`: hướng dẫn chạy.
