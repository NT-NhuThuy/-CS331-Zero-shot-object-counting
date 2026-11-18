# Gói phát triển method T1 của Thúy — UNICOUNT

**Ngày:** 2026-11-18
**Phạm vi:** identity-aware joint arbitration của Thúy
**Trạng thái:** experimental method / CPU-tested / chưa có real-data claim

## 1. Bài toán

CountGD++ tạo candidate box cho một ảnh và một query. Candidate có thể là:

- `baseline`: proposal từ ảnh chính;
- `rescue`: proposal tìm thêm từ tile hoặc view khác.

Hai box gần nhau chưa chắc là cùng một vật thể. Chúng có thể là cùng một
physical instance, hai vật thể khác nhau bị overlap, hoặc một trường hợp chưa
đủ bằng chứng.

Pipeline cũ match rescue với baseline bằng IoU trước. Nếu IoU cao, rescue bị
coi là duplicate và bị bỏ qua. Vì vậy một vật thể thật có thể bị mất khi
baseline và rescue thực ra là hai instance khác nhau.

## 2. Method đã phát triển

Gói này thêm nhánh experimental `joint_t1`, chạy song song với
`sequential_v1`.

### `sequential_v1`

Đây là baseline cũ và vẫn là default:

```text
unary verification → IoU match → identity graph → final count
```

### `joint_t1`

```text
frozen baseline + rescue candidates
        ↓
unary evidence: YES / NO / UNKNOWN
        ↓
pairwise identity: SAME / DIFFERENT / UNKNOWN
        ↓
rescue component clustering
        ↓
baseline–rescue arbitration
        ↓
KEEP / ADD / COALESCE / REJECT / UNRESOLVED / ABSTAIN
```

Luật chính:

- `SAME`: cùng physical instance, coalesce;
- `DIFFERENT`: giữ như hai physical instances;
- `UNKNOWN`: không được tự nâng thành `SAME`, `DIFFERENT` hoặc `NO`;
- rescue overlap baseline chỉ được add khi collision relations cần thiết đều
  là `DIFFERENT`;
- rescue `SAME` với baseline được ghi chung một physical component;
- conflict hoặc thiếu evidence được ghi `UNRESOLVED`;
- query có thể abstain nhưng fallback count vẫn được giữ riêng.

## 3. File code chính

### Core method

- `src/countlab/method/pipeline.py`: chọn `sequential_v1` hoặc `joint_t1`, giữ
  input hash, policy và flags nhất quán.
- `src/countlab/method/joint_arbitration.py`: baseline–rescue arbitration,
  pairwise evidence, physical partition và safety invariants.
- `src/countlab/method/identity_graph.py`: complete-link graph; không dùng
  geometry để giả lập identity evidence.
- `src/countlab/method/pairwise_allocation.py`: chọn pairwise query theo
  `impact`, `geometry` hoặc `risk` mà không nhìn trước relation answer.

### Controls và evaluation

- `src/countlab/method/baselines.py`: controls B0–B9.
- `src/countlab/method/t1_comparison.py`: matched comparison trên cùng input.
- `src/countlab/method/t1_replay.py`: create-only replay, identity metrics và
  count metrics offline.
- `src/countlab/method/t1_batch_replay.py`: batch replay có hash artifact.
- `src/countlab/method/t1_batch_evaluate.py`: offline MAE/exact, abstention,
  false merge, false split và missed same.
- `src/countlab/method/joint_diagnostics.py`: giải thích query/action thay đổi
  so với v1.

## 4. Chạy code

### Test

```bash
cd /mlcv1/WorkingSpace/Personal/core_baotg/thuy/UNICOUNT
PYTHONPATH=src .venv/bin/pytest -q
```

Kết quả lúc đóng gói: `114 passed`.

### Toy batch replay

```bash
cd /mlcv1/WorkingSpace/Personal/core_baotg/thuy/UNICOUNT
tmp_root=$(mktemp -d /tmp/unicount-t1-toy.XXXXXX)
cp fixtures/unified_toy.json "$tmp_root/manifest.json"
python3 -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"schema":"countlab_t1_manifest_index_v1","manifest_paths":["manifest.json"]})+"\n")' "$tmp_root/index.json"
PYTHONPATH=src .venv/bin/python -m countlab.method.t1_batch_replay \
  --index "$tmp_root/index.json" \
  --config configs/experiments/t1_joint_replay_v1.json \
  --out-dir "$tmp_root/out"
```

Toy replay chỉ kiểm tra pipeline, không phải kết quả paper.

### Offline evaluation

```bash
PYTHONPATH=src .venv/bin/python -m countlab.method.t1_batch_evaluate \
  --index /path/to/manifest_index.json \
  --annotations-index /path/to/annotations_index.json \
  --config configs/experiments/t1_joint_replay_v1.json \
  --out /path/to/t1_offline_evaluation.json
```

Annotation index dùng schema `countlab_t1_annotation_index_v1`. Mỗi
`image_id/query_id` phải map tới đúng một annotation file có:

```json
{
  "schema": "countlab_t1_instance_truth_v1",
  "image_id": "...",
  "query_id": "...",
  "ground_truth_count": 2,
  "truth_by_candidate": {"candidate_id": "reviewed_instance_id"}
}
```

Ground truth chỉ được join sau khi mọi arm đã replay xong.

## 5. Kết quả hiện có

| Case | v1 | joint T1 | Ý nghĩa |
|---|---:|---:|---|
| overlap + `DIFFERENT` | 1 | 2 | rescue đúng được add |
| overlap + `SAME` | 1 | 1 | không double-count, không false split |
| overlap + `UNKNOWN` | 1 | 1 | không tự promotion |

Đây là synthetic logic validation, chưa phải real-data improvement.

## 6. Chưa được claim

Chưa được claim SOTA, novelty, real-data MAE gain, source-disjoint
generalization, certified count bounds hoặc universal false-rejection safety.

Các dependency còn thiếu:

1. P02 frozen candidate manifest với stable candidate IDs;
2. pairwise verifier evidence/cost thật;
3. P03 approved source-disjoint DEV/CAL split;
4. reviewed instance-level annotations;
5. untouched final evaluation.

## 7. Cập nhật sang local repo an toàn

Không clone đè hoặc reset repo local. Trước khi copy/cherry-pick thủ công:

```bash
git status --short
git branch --show-current
git diff --stat
```

Không chạy `git reset --hard`, `git clean -fd`, `git checkout .` và không commit
toàn bộ worktree nếu trong đó có file thuộc Lộc/Thư hoặc thay đổi cũ chưa review.

## 8. Kết luận

Đây là snapshot method branch có joint arbitration, identity safety rules,
matched controls, GT-separated replay/evaluation và CPU regression suite. Bước
tiếp theo là đưa P02/P03 artifacts vào replay, đo trên fresh DEV/CAL, so sánh
với B6/B8 ở cùng candidate pool/evidence budget, rồi mới quyết định có cần GPU
và có đủ headroom cho paper claim hay không.
