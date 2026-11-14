# UNICOUNT — TEAM START HERE (2026-11-14)

> **One method, three module owners, one shared experimental protocol.** Đây là hướng dẫn vận hành repository cho Thư, Lộc, Thúy. Đọc cùng `README.md`, `docs/method/INTERFACE_CONTRACT.md`, `docs/verification/UNIFIED_RESEARCH_AUDIT_20261114.md`. *A* và SOTA là mục tiêu nghiên cứu, không phải kết quả hiện có.*

## 0. Trạng thái và điều kiện tiên quyết

- **Branch làm việc:** `research/unified-integration-verified` (KHÔNG dùng tên branch cũ `research/unified-method-v1` trong playbook trước đây).
- **Hiện có:** một schema, unified controller, offline replay, batch replay, online entrypoint, preflight; 75 CPU tests passed trên clean clone tại commit `5ad4ead`. Khi tài liệu này được commit, chạy lại CI.
- **Chưa xác thực:** CountGD++ official pseudo-exemplar `num_pseudo_exemplars=3` parity với V14; Qwen GPU smoke và calibration; cross-dataset learned risk head; raw `full_cohort_output.json` V14; GPU end-to-end MAE của *unified* method.
- **Không đánh đồng:** `countlab-method` = frozen evidence replay; `countlab-online` = code path gọi model thật **chỉ khi** đủ checkpoint và dependencies. `configs/integrations/native_single_pass_example.json` là operating point **NO PSEUDO**, không tương đương V14.
- `legacy/{thu,loc,thuy}` giữ read-only để so sánh, không tiếp tục phát triển pipeline riêng trong `legacy/`.

## 1. Team lead: publish branch lần đầu từ bundle

Chạy trên máy có Git và quyền push repo `https://github.com/NT-NhuThuy/-CS331-Zero-shot-object-counting.git`. Nếu GitHub chưa có branch mới, thành viên **chưa thể** `git switch --track` branch đó từ origin.

```powershell
git clone "https://github.com/NT-NhuThuy/-CS331-Zero-shot-object-counting.git" cs331-unicount
cd cs331-unicount
# Đổi đường dẫn bundle theo nơi đã tải. KHÔNG dùng bundle được clone làm remote origin.
git fetch "C:\Users\YOU\Downloads\CS331_UNICOUNT_TEAM_READY_ALL_BRANCHES_20261114.bundle" "refs/heads/research/unified-integration-verified:refs/heads/research/unified-integration-verified"
git switch research/unified-integration-verified
git push -u origin research/unified-integration-verified
```

Lệnh `git push` **chỉ tạo/cập nhật branch nghiên cứu** trên GitHub; không push `main`. Team lead tạo branch protection, yêu cầu >=1 review và test trước merge. Nếu branch đã được push thì bỏ qua bước import bundle.

## 2. Mỗi thành viên: clone, tạo môi trường, kiểm thử

### Windows PowerShell

```powershell
git clone "https://github.com/NT-NhuThuy/-CS331-Zero-shot-object-counting.git" cs331-unicount
cd cs331-unicount
git fetch origin
git switch --track origin/research/unified-integration-verified
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest -q tests/ legacy/thuy/branch/ legacy/loc/tools/praco_test1/test_verify_praco_artifacts.py
.\.venv\Scripts\python.exe -m countlab.method.cli --input fixtures/unified_toy.json --config configs/method/unicount_v1.json --out local_toy_result.json
```

**Chú ý PowerShell:** dùng đường dẫn thư mục `legacy/thuy/branch/` thay vì glob `test_*.py` để pytest tự tìm test files. `local_toy_result.json` phải chưa tồn tại: output là create-only.

### Linux / server

```bash
git clone 'https://github.com/NT-NhuThuy/-CS331-Zero-shot-object-counting.git' cs331-unicount
cd cs331-unicount
git fetch origin
git switch --track origin/research/unified-integration-verified
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q tests/ legacy/thuy/branch/ legacy/loc/tools/praco_test1/test_verify_praco_artifacts.py
countlab-method --input fixtures/unified_toy.json --config configs/method/unicount_v1.json --out local_toy_result.json
```

Nếu lệnh Python/venv lỗi, báo version OS, Python, trace và package lock; không tự nâng/hạ model dependencies trong PR không liên quan. Không commit `local_toy_result.json`, checkpoint, image raw hoặc credentials.

## 3. Một pipeline — ai sửa file nào?

```text
image+query
  → CountGD++ native (shared / Lộc) → baseline boxes
                         └→ tiled proposal recovery (Lộc) → rescue boxes
  → candidate IDs + original coordinates + provenance
  → GT-blind risk routing, budget (Thư)
  → Qwen region semantic / whole evidence (Lộc)
  → optional identity pair evidence (Lộc → Thúy)
  → identity-consistent bidirectional reconciliation (Thúy)
  → count, accepted/rejected/UNKNOWN, evidence trace
  → post-inference offline evaluation (shared, GT separate)
```

**Thư — router / calibrated risk:**
- Đọc `src/countlab/method/router.py`, `src/countlab/integrations/thu_features.py`, `legacy/thu/scripts/p14_08b_09_joint_unified.py`, `tests/test_thu_features.py`.
- PR đầu: `feat/thu-split-audit` — source-image-disjoint TRAIN/DEV/CAL/FINAL_SEALED manifest, tests chống overlap, dataset registry; không dùng FSC TEST đã exposed làm fresh test.
- PR kế: `feat/thu-risk-features` — exact candidate mapping, feature extraction ở runtime tương thích, risk/calibration fit chỉ TRAIN/CAL; benchmark vs random, confidence và margin theo **matched Qwen calls/tokens/latency** ở ngân sách 25/50/100%.
- **Acceptance:** không đọc GT khi inference; không dùng NPZ FSC frozen trên LVIS không khớp; 1 fixture + tests + split/hash/calibration manifest + reliability/utility plot. Learned scalar Joint V3.2 giữ comparator; KHÔNG tự cộng một scalar vào `count` sau pipeline.

**Lộc — backend / recovery / region verifier:**
- Đọc `src/countlab/integrations/{countgd_native,recovery,qwen_unary,online,preflight}.py`, `configs/integrations/native_single_pass_example.json`, `legacy/loc/notebooks`, `tests/test_online_integration.py`, `tests/test_preflight.py`.
- PR đầu: `feat/loc-countgd-adapter` — official CountGD++ **pseudo=3** adapter riêng; lưu source/checkpoint/BERT/prompt/threshold hashes, so sánh box/count parity với operating point V14. KHÔNG âm thầm sửa native NO-PSEUDO adapter.
- PR kế: `feat/loc-region-verifier` — pin Qwen model revision, prompts/response parser, `YES/NO/UNKNOWN` semantic và whole-instance ở **region**, failure/latency/calls; global V7.6 dual-NO là comparator riêng.
- PR kế: `feat/loc-multiview-recovery` — mapping tile boxes về original image, consensus, recall/false-candidate/cost, so với single/full + raw tile union.
- **Acceptance:** ≥1 GPU smoke fixture với đúng checkpoint, original-coordinate parity, parser UNKNOWN trên lỗi, hash+latency/VRAM, case gallery và regression/false-negative audit. Không claim V14 parity trước khi kiểm chứng.

**Thúy — FQIE provenance / identity / arbitration:**
- Đọc `src/countlab/method/{schema,identity,pipeline,verifier}.py`, `src/countlab/adapters/fqie_legacy.py`, `legacy/thuy/branch/{safe_iic,bidirectional_evidence_controller}.py`, `tests/test_unified_method.py`.
- PR đầu: `feat/thuy-v14-import` — thu hồi V14 `full_cohort_output.json`, immutable hashes/splits/raw evidence, tái tính all/present/absent MAE, sign-test và source-cluster bootstrap; audit freeze-before-test.
- PR kế: `feat/thuy-identity-adapter` — relation `SAME/DIFFERENT/UNKNOWN`, correct original coords, no UNKNOWN merge, no unsupported transitive bridge, false merge/split và instance-recall metrics.
- PR kế: `feat/thuy-evidence-arbitration` — đối chiếu simple NO/NO + IoU/NMS, complete-link, Union-Find, independent bidirectional; đề xuất solver mới **chỉ khi** G1 đã chứng minh reviewer-relevant gap.
- **Acceptance:** full trace của accepted/rejected/UNKNOWN, explicit abstention and baseline fallback, frozen policy hashes, objective+constraint+complexity, correctness/ablation trên fresh DEV và safety checks.

## 4. Cách tạo PR (mỗi người làm giống nhau)

```bash
git switch research/unified-integration-verified
git pull --ff-only origin research/unified-integration-verified
git switch -c feat/thu-split-audit  # Đổi thành feat/loc-countgd-adapter hoặc feat/thuy-v14-import
# Sửa src/countlab/ + tests/ + configs/ + docs/, legacy chỉ tham chiếu
python -m pytest -q tests/ legacy/thuy/branch/ legacy/loc/tools/praco_test1/test_verify_praco_artifacts.py
git status
git add src/countlab tests configs docs
git commit -m "feat: add source-disjoint split audit"
git push -u origin feat/thu-split-audit
```

Mở GitHub Pull Request **base = `research/unified-integration-verified`**, compare = feature branch. Reviewer kiểm tra schema, GT leakage, tests, protocol SHA, cost, negatives. Sau merge, mọi người `git switch research/unified-integration-verified && git pull --ff-only` trước khi tạo PR khác.

Một PR hoàn chỉnh bắt buộc có: WHY (failure/hypothesis), WHAT (code paths), HOW TESTED, protocol ID & input hash, metrics vs matched baseline/CI, negative cases, compute budget, changed outputs, reproducibility command. Không thay `legacy`/frozen results và không force-push history chung.

## 5. Future plan với cửa nghiệm thu (không chạy theo lịch thời gian cứng)

| Gate | Cần làm | Owner | Điều kiện GO |
|---|---|---|---|
| **G0 / Foundation** | Official pseudo=3 baseline, V14 raw parity, splits TRAIN/DEV/CAL/FINAL_SEALED, Qwen revision, raw hashes | Lộc + Thúy + Thư | Full baseline predictions match frozen protocol theo tolerance khóa trước, không GT leakage; không dùng V14 làm tuner |
| **G1 / Mechanism** | Dual-NO region verifier vs threshold-calibrated baseline, semantic-only/whole-only, Qwen global veto; false rejection và class-mismatch | Lộc + Thúy | Có lợi ích thực tế trên DEV độc lập; safety margin được khóa trước; nếu không, giữ dual-NO làm baseline, không claim novelty |
| **G2 / New algorithm** | Router 25/50/100% cost vs random/margin; graph arbitration vs NMS/Union-Find/complete-link; uncertainty policy | Thư + Thúy; Lộc evidence | Target nghiên cứu: ≥90% gain all-verify ở ≤50% calls; ≥20% false-merge reduction, ≤2 pp instance recall loss — **targets, chưa phải kết quả** |
| **G3 / Generalization** | New LVIS source-disjoint; multi-category PrACo++/MUCCA-compatible setting; additional domain with adequate instance annotations | Cả nhóm | ≥2 independent evaluation regimes, matched-cost reporting, clustering by source, negative/abstention and safety metrics, no claim beyond evidence |
| **G4 / Paper & SOTA attempt** | DAVE/ValidCounter/CountSE and other compatible baselines, ablations, latency/VRAM, manuscript/figures | Cả nhóm | A novel technical mechanism + controlled improvement + reproducibility; only call SOTA if matched public benchmark actually supports it |

**Critical stop:** nếu G1 chỉ ngang threshold calibration, cần nghiên cứu lại hypothesis trước G2; nếu G2 không thắng simple graph matching thì bỏ novelty claim về arbitration. FSC147 TEST, PrACo Test-1, V14 đã exposed trong dự án, không biến thành new confirmatory set.

## 6. Đường chạy GPU — chỉ sau G0 readiness

```bash
# Server GPU: separate environment, install online deps only if compatible with official upstream
python -m pip install -e '.[test,online]'
# Tạo bản copy riêng của configs/integrations/native_single_pass_example.json
# Thay repo/checkpoint/bert_dir bằng đường dẫn local, Qwen revision bằng exact pinned commit
countlab-preflight --config configs/integrations/native_single_pass_example.json
countlab-online --image /real/path/demo.jpg --image-id demo --query-id q1 --query chair \
  --config configs/integrations/native_single_pass_example.json --out-dir outputs/demo_v1
```

`countlab-preflight` trả **BLOCKED** khi config còn placeholders hoặc package thiếu là hành vi đúng. `outputs/demo_v1` không được tồn tại trước khi chạy. Có thể chỉ chạy smoke với native NO-PSEUDO diagnostic config này; **không đưa số đó vào bảng parity V14**. Nếu cần parity pseudo=3, trước tiên phải phát triển adapter riêng ở PR của Lộc.

## 7. Team meeting / reporting tối thiểu

Mỗi PR report 1 đoạn: mục tiêu, nhánh/commit, input cohort & source, model/prompt/version, metric và 95% CI, oracle/GT-separation, runtime/VRAM, failure examples, GO/HOLD/NO-GO. Duy trì `paper/claim_ledger.csv` (claim → code commit → dataset → experimental evidence → trạng thái). Một người merge owner, một người review độc lập; thay đổi schema/policy bắt buộc review cả ba.

## 8. Reading order nhanh

1. `README.md` (trạng thái)
2. `docs/TEAM_START_HERE.md` (đây)
3. `docs/method/INTERFACE_CONTRACT.md` (schema, rule GT-blind)
4. `docs/verification/UNIFIED_RESEARCH_AUDIT_20261114.md` (kết quả, blockers, novelty)
5. `docs/method/PRIORITIZED_TASKS.md` (12 PR phụ thuộc)
6. `docs/method/MILESTONES.md` (M0→M5)

**Một định nghĩa DONE duy nhất:** PR merged + tested + reproducible artifact + correct protocol + independent matched controls. `GPU inference worked` chỉ là integration milestone, **chưa** chứng minh paper contribution.
