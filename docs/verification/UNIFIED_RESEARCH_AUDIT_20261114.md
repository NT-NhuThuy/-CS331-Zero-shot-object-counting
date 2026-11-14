# UNICOUNT — kiểm toán lại code, contribution, novelty và lộ trình A* (14/11/2026)

**Kết luận:** Đã xây dựng **một pipeline phần mềm có thể gọi các backend thật**, nhưng **chưa có kết quả GPU hoặc checkpoint parity của UNICOUNT tích hợp**. Ba nhánh có bằng chứng thực nghiệm thuộc **ba protocol khác nhau**. V14 là kết quả được báo cáo và có bốn artifact summary khớp hash, **không phải kết quả của pipeline UNICOUNT mới**. A* và SOTA là **mục tiêu**, chưa thể coi là claim.

## 1. Phạm vi kiểm toán và tính chính xác của kết luận

Nguồn: Git branch gốc, `legacy/thu`, `legacy/loc`, `legacy/thuy`, source Python + notebooks, results CSV/ZIP, `FQIE_BIDIRECTIONAL_HOLDOUT_RESULTS_20261116.md`, CountGD++ official repo/paper, DAVE, ValidCounter, CountSE, PrACo++, SAM3Count, AdaCount, GeCo2. Tất cả sản phẩm mới ở branch `research/unified-integration-verified`.

Phân biệt bốn trạng thái:

- **AUDITED_ARTIFACT:** nội dung/hash và metric từ file thật được kiểm tra; không đồng nghĩa chạy lại model.
- **EXECUTABLE_CPU:** code chạy qua unit/integration tests trên CPU; mock backend không chứng minh độ chính xác model.
- **GPU_READY_CONDITIONAL:** có đường code thật và interface, nhưng cần upstream/model/checkpoint/GPU, cần smoke/parity; không được claim reproducible trước khi chạy.
- **RESEARCH_PROPOSED:** hypothesis/method mới chưa implement/chưa chứng minh.

### Input và nguồn đã xác nhận

- Lịch sử Git các nhánh gốc: `main f8cfe68`, `feature/thu-joint-semantic-counting c945ea5`, `loc/research-v4-v75-2026-11-06 86aefa4`, `loc/praco-official-adapter-v76-full 6c86d30`; branch tích hợp cũ `research/unified-method-v1 8e52d36`.
- `legacy/thuy/src/zoc/backends/countgd_grounding/backend.py`: có CountGDBackend, đọc repo upstream + checkpoint + BERT từ local và chạy native single-pass, output normalized boxes/scores/native_text_pass; **không chạy pseudo-exemplar feedback** trong path này.
- `legacy/loc/notebooks/01_original_pipeline/V4_Qwen3_VL_Verification.ipynb`, `04_validation/V7_5_Blind_Validation_240.ipynb`: có Qwen3-VL processor+generate và nhiều thuật toán multi-scale. `V7.6` global dual-NO veto **không bằng** region semantic+whole checker.
- `legacy/thu/scripts/p14_08b_09_joint_unified.py` và `frozen_v4_a3_unified_nodes.npz`: learned Joint V3.2 count corrector và V4/A3 node features trên FSC 1301 pairs; **không có trained checkpoints và đầy đủ graph/decoder caches trong repo**.
- `legacy/thuy/branch/bidirectional_evidence_controller.py`, `safe_iic.py`, `prepare_bidirectional_verification.py`: GT-free YES/NO/UNKNOWN policies, IoU match, conservative association.

## 2. Method chính: chính xác đã nối gì vào chỗ nào?

```text
IMAGE + CATEGORY TEXT
   |
   v
[A] CountGD++ native backend -- Thúy source adapter (actual optional PyTorch execution)
   | baseline boxes + native score + original coordinates
   +------------------------+
   |                        |
   v                        v
[B] Thư reliability /      [C] Lộc tile candidate recovery
    risk router              (actual native backend crop inference)
    exact-key V4/A3         | rescues with view IDs, original coords
    optional frozen map     |
   +------------+-----------+
                v
[D] Budgeted selected candidates -> Qwen3-VL REGION semantic and whole checks
    (Lộc extraction with NEW frozen prompts; 2 GT-free calls per selected box)
                v
[E] Optional Qwen pair identity edges (SAME/DIFFERENT/UNKNOWN), no IoU-as-proof
                v
[F] Thúy bidirectional NO/NO rejection + YES/YES unmatched rescue
    + conservative complete-link identity / ambiguity handling
                v
  count + accepted/rejected/unknown + provenance + limited observed-candidate interval
                v
  separate GT-only evaluation after inference output is sealed
```

### Thực sự đã implement ở root

| Module | Main file | Legacy source / reuse | Runtime status | Được phép claim |
|---|---|---|---|---|
| [A] Native baseline | `src/countlab/integrations/countgd_native.py` | Lazy-import trực tiếp `legacy/thuy/src/zoc/backends/countgd_grounding/backend.py` | GPU_READY_CONDITIONAL; single-pass only | Source wired, **không** V14 parity |
| [B] Risk/features | `src/countlab/integrations/thu_features.py` + `method/router.py` | Reads exact-key NPZ V4/A3, calibration optional | EXECUTABLE_CPU; source-specific FSC | Data join + router work; learned Joint V3.2 **chưa** thành universal router |
| [C] Recovery | `integrations/recovery.py` | Multi-grid inspired by Lộc + Thúy; uses real native model per tile | GPU_READY_CONDITIONAL | Distinct **NEW** protocol, **không** parity Lộc V7.5 hoặc Thúy V14 |
| [D] Unary Qwen | `integrations/qwen_unary.py` | Uses Qwen3VL native generator process from Lộc notebooks | GPU_READY_CONDITIONAL | Actual model calls implemented, prompt NEW, không tuyên bố reproduce V7.6 |
| [E] Pair Qwen | `integrations/qwen_unary.py::verify_identity_pairs` | New online bridge; GT-free two regions | GPU_READY_CONDITIONAL | Pairwise evidence path exists; **chưa** calibrated/validated |
| [F] Reconcile | `method/{schema,identity,pipeline}.py` | Transfers principles and code logic from Thúy FQIE | EXECUTABLE_CPU | Deterministic software-level invariants only |
| One CLI | `integrations/online.py::main` → `countlab-online` | All above | GPU_READY_CONDITIONAL | Online entrypoint exists, **không** benchmark inference done |
| Replay CLI | `method/cli.py` → `countlab-method` | precomputed evidence | EXECUTABLE_CPU | Replay output; không phải model accuracy |

### Vì sao nói “chưa nối hết” trước đây và đã sửa những gì?

Bản trước chỉ truyền candidate đã có YES/NO và dùng một `PrecomputedUnaryVerifier`. Nó **không gọi** CountGD++ hay Qwen, không đọc feature NPZ của Thư. Nay có `countlab-online`: gọi `NativeCountGD` thật, crop-based recovery, Qwen region semantic/whole, Qwen pairwise identity tùy chọn và final FQIE controller. Đã thêm exact-key adapter cho feature V4/A3 có thể thay đổi thứ tự chọn candidate với calibration riêng. Các module đã được nối về **cùng một graph chạy xuyên suốt**.

**Còn thiếu để trở thành official unified method:** không có đầy đủ pretrained weights, Qwen snapshot/pinned revision local, CountGD++ upstream + BERT, V14 raw source data, exact Lộc V7.5 crop/multi-scale protocol parity, trained Joint V3.2 cross-dataset router. Bản native chỉ một pass, nên phải thêm official CountGD++ pseudo-exemplar adapter riêng cho operating point của V14. Không thể hứa chỉ đổi tên folder là ba method sẽ cộng được số liệu.

## 3. Metric audit — KHÔNG gộp protocol khác nhau

### Thúy V14 LVIS-derived: reported HOLDOUT, summary hashes verified

| 180 pairs; 90 present + 90 absent | Frozen CountGD++ | Bidirectional FQIE |
|---|---:|---:|
| MAE tổng | 1.761111 | 1.100000 |
| Present MAE | 3.355556 | 2.200000 |
| Absent MAE | 0.166667 | 0.000000 |
| Exact count accuracy | 56.67% | 60.56% |
| Bias | +0.650000 | −0.055556 |

- Relative ΔMAE `−37.54%`, 22 cải thiện / 3 xấu đi / 155 hòa. Paired sign-test p `0.00015652`, source-clustered bootstrap CI reported `[-1.3833,-0.1833]` (10,000 resamples); chưa rerun cluster bootstrap từ per-query raw records.
- V14 ablation: baseline `1.7611`, rescue-only `1.7444`, rejection-only `1.1167`, full `1.1000`. Approximately **97.5% of full-vs-baseline MAE reduction** is achieved before the rescue addition. Đây là novelty risk: cần thắng calibrated score threshold / comparators.
- V13 development: `4.1167 → 2.8944`, 32 improved / 1 worsened; dev không phải independent confirmation của newly selected method.
- **Đã kiểm tra hash khớp chính xác:** `offline_metrics.json e553a87b...`, `paired_audit.json 9d250bdc...`, `ablation_metrics.json 39d437ca...`, `failure_audit.json 4fd8fe29...`. `full_cohort_output.json` SHA ghi `94e71b3f...` nhưng **file không có trong mirror**. Do đó chưa thể reproduce metric trực tiếp từ raw 180 predictions hoặc kiểm tra từng inference evidence. `20261116` là mốc *sau* thời điểm kiểm toán 14/11/2026; phải giải thích timestamp/provenance, không nhận là đã freeze hợp lệ nhờ tên file.

### Thư FSC-147 unified 1301 pairs (1190 present, 111 hard-neg)

CPU đọc trực tiếp `legacy/thu/outputs/joint_semantic_counting_v1/p14_08b_09_joint_unified_frozen_v1/unified_per_pair_predictions.csv` rồi tính MAE từ `gt_count`:

| Arm | Recomputed MAE |
|---|---:|
| CountGD | 15.990776 |
| V4 | 15.230338 |
| Joint V3.2 | 14.508626 |
| Joint V3.2 + A3 | 13.228046 |

Dataset có FSC-147 TEST đã được quan sát/khai thác, nên là exploratory, **không** re-label thành untouched holdout. V3.2 là learned count correction scalar, không tự chuyển thành physical-instance router. NPZ V4/A3 shape `(88432,)` là frozen nodes, mapping bằng `(node_graph_id, raw_query_index)`.

### Lộc PrACo Test-1 official A/B

Đã chạy `legacy/loc/tools/praco_test1/verify_praco_artifacts.py` trên hai ZIP gốc: **PASS**, 34,510 pairs / 1,190 ảnh / 29 classes, 4,303 Qwen calls, 3,060 suppressions (3,041 class-mismatch, **19 true-positive-class**). Mismatch-nonzero `9.342737% → 0.216086%`, positive MAE **worse** `12.180672 → 13.528571`, positive zero `0 → 19`. Đây là trade-off thực, không được claim được nâng MAE cùng setting.

**Không thể so các MAE của V14, FSC và PrACo như một leaderboard**: khác source, input prompt, pseudo-exemplars, supervision, số query, tính độc lập test.

## 4. Technical risks ngay trong core (đã nhận diện, chưa chữa bằng số liệu)

1. **Dual-NO logic:** `semantic=NO, whole=YES` không reject; bảo thủ nhưng có thể giữ confusers. `semantic=YES, whole=NO` không reject part boxes. Tại sao cần cả hai NO phải được chứng minh bằng cost-sensitive calibration/ablation.
2. **Baseline duplication chưa giải quyết:** Identity graph mới xử lý rescues, baseline boxes trùng vẫn có thể đếm nhiều nếu không NO/NO.
3. **IoU≥0.5 baseline-rescue** là match heuristic, không chứng minh cùng physical identity; có thể bỏ sót hai vật thể thật chồng nhau.
4. **Greedy complete-link** phụ thuộc thứ tự và không giải joint global optimum; bridge/UNKNOWN handling có thể hy sinh recall.
5. **UNKNOWN baseline vẫn cộng vào point count** dù accepted set không chứa chúng; output `count_status=partial`, lower/upper chỉ nói về observed candidates, không phải interval bảo đảm cho scene ground truth.
6. **Qwen unary hai prompt cùng backbone** không phải hai nguồn bằng chứng thống kê độc lập. Parse nghiêm ngặt chuyển câu giải thích thành UNKNOWN; cần đo unknown/failure fraction.
7. **Budget router hiện chưa calibrated:** nếu không có calibration và exact-key nodes thì chỉ ưu tiên theo `1-confidence`/risk hint. Đây là kỹ thuật screening, không phải learned risk controller.
8. **Online native protocol khác V14:** single-pass/no-pseudo vs V14 frozen (`num_pseudo_exemplars=3`). Không chuyển Δ37.5% sang UNICOUNT.
9. **Không có chứng nhận source-disjoint từ summary alone:** cần V14 split manifest, run timestamp + hash + model rev + upstream code + raw outputs.
10. **PrACo official adapter** đã có negative signal; không đưa global dual-NO image veto làm mặc định core.

## 5. Novelty audit — so với literature nguyên thủy

| Chúng ta định claim | Prior art gần nhất | Nhận xét về novelty hiện tại |
|---|---|---|
| Detection → verification → filter errors | DAVE CVPR 2024; ValidCounter 2025 | **Không mới riêng lẻ.** V14 outcome tốt nhưng rejection-only chính là core của nhiều works. |
| Negative/positive prompting & pseudo-exemplars | CountGD++ CVPR 2026 | **Đã có ở backbone.** |
| Multi-scale tile/recovery | CountGD++ adaptive cropping; CountSE ICCV 2025; SAM3Count CVPRW 2026; AdaCount 2026 | **Không mới riêng lẻ.** |
| Region-level category evidence | ValidCounter, DAVE, semantic exemplar filtering | Cần contribution về evidence arbitration, không chỉ gọi Qwen. |
| Identity graph deduplication | NMS and clustering, multi-view reconciliation | Complete-link clique/UNKNOWN là conservative heuristic phổ biến; chỉ code invariants chưa đủ A*. |
| Auditable provenance and abstention | Broad selective prediction / traceability literature | Giá trị protocol nhưng **không đủ** novelty độc lập nếu thiếu meaningful verification guarantee/metric. |
| **Joint risk-budgeted evidence/identity arbitration** | Các hướng trên làm từng phần, nhưng cần literature search sâu hơn | **Research hypothesis**, hiện chưa có novelty proof, theorem hay benchmark gains. |

Primary sources:
- CountGD++ official https://www.robots.ox.ac.uk/~vgg/publications/2026/AminiNaienib26/ ; https://github.com/niki-amini-naieni/CountGDPlusPlus
- DAVE CVPR 2024 https://openaccess.thecvf.com/content/CVPR2024/html/Pelhan_DAVE_-_A_Detect-and-Verify_Paradigm_for_Low-Shot_Counting_CVPR_2024_paper.html
- ValidCounter 2025 https://www.sciencedirect.com/science/article/pii/S0020025525002932
- CountSE ICCV 2025 https://openaccess.thecvf.com/content/ICCV2025/html/Liu_CountSE_Soft_Exemplar_Open-set_Object_Counting_ICCV_2025_paper.html
- PrACo++ / MUCCA 2026 https://arxiv.org/abs/2605.02752
- SAM3Count CVPR Workshops 2026 https://github.com/Joan947/SAM3Count
- AdaCount 2026 https://arxiv.org/abs/2607.02139
- GeCo2 AAAI 2026 https://ojs.aaai.org/index.php/AAAI/article/view/37780

**Phán quyết:** V14 result có ý nghĩa empirical trong protocol của nó; **current UNICOUNT chỉ là một integrated research baseline**, chưa đủ method-level novelty/SOTA. Giữ claim A* ở trạng thái **NO-GO FOR SUBMISSION** cho đến khi có một cơ chế joint arbitration mới và đối chứng matched-budget.

## 6. Primary hypothesis A* cần cùng kiểm nghiệm

**Working question:** *Can a frozen open-world counter correct spurious and missing instances through joint evidence-consistent physical-instance selection under explicit risk and verification budgets?*

Giữ các module I/O hiện nay nhưng nghiên cứu một arbitration engine mới: candidate graph gồm cả baseline và recovered proposals; tập evidence cho mỗi candidate gồm semantic/whole/uncertainty; identity edges có SAME/DIFFERENT/UNKNOWN; router hỏi verifier nào dựa trên expected benefit và call cost, giải quyết tranh chấp toàn cục thay vì cộng/trừ counts. Có thể nghiên cứu constrained optimization hoặc correlation clustering có abstention; ưu tiên định nghĩa objective, constraint, assumptions và computational complexity **trước** khi gọi nó novelty. Đặc biệt cần kiểm nghiệm rằng thuật toán tối ưu mang lợi ích vượt heuristic thresholding + NMS ở matched budget. Nếu không, bỏ nhánh method mới để tránh claim quá đà.

Đóng góp **đề xuất** (chưa phải claim):

1. **Joint evidence-consistent assignment:** xử lý baseline + rescue + identity/part/UNKNOWN cùng một bài toán, thay vì separate hard gates.
2. **Calibrated asymmetric risk:** giảm false positives với upper-bound/coverage controls đã được calibration hợp lệ; không tuyên bố statistical guarantee trước khi chứng minh assumptions.
3. **Budget-aware verification:** phân bổ model calls tới các candidate có expected utility cao hơn margin/random tại cùng ngân sách.
4. **Instance-level evidence outputs:** accepted/rejected/abstained sets với provenance, không gọi full-scene exact khi UNKNOWN còn lại.

Nếu C1 chỉ tương đương DAVE+NMS/heuristic; C2 không được calibration và C3 không thắng margin ở matched calls, **method không đạt cổng A*** dù MAE V14 vẫn đẹp.

## 7. Kế hoạch thực nghiệm có GO/NO-GO và target đã định nghĩa trước

| Gate | Chủ trì | Thí nghiệm và baseline bắt buộc | Acceptance target (đặt trước, **không phải kết quả**) | Nếu fail |
|---|---|---|---|---|
| **G0: source/GPU parity** | Lộc + Thúy | Real CountGD++ official with pseudo=3 (V14) and no-pseudo native; V14 raw artifacts; Qwen model revision, hashes, 20–50 GT-free pilot pairs | 100% matching per-query baseline boxes/counts within predeclared floating tolerance; no inference GT; freeze inputs/config/SHA | STOP all metric comparisons |
| **G1: rejection is nontrivial** | Lộc + Thúy | On fresh DEV/CAL: calibrated CountGD threshold, dual-NO, sem-only, whole-only, Qwen global veto, Qwen region | Significant/count-robust benefit beyond threshold on independent holdout, and no unacceptable false rejections; predeclare safety margin | If no separation, dual-NO only remains baseline |
| **G2: risk router** | Thư | Train clean risk model on train only; compare 25/50/100% call budgets vs random, score, margin and all-verify | Aspirational: ≥90% all-verify gain at 50% verification calls on DEV, no safety failure; paired CIs | Fall back deterministic routing |
| **G3: identity/arbitration** | Thúy | New graph engine vs NMS, Union-Find, simple complete-link, independent bidirectional | Aspirational: ≥20% relative false-merge reduction without >2 pp instance recall loss on DEV; MAE/risk–coverage better at matched cost | If not, avoid claiming graph novelty |
| **G4: independent external evaluation** | Shared | LVIS-derived fresh source-disjoint; new multi-category (MUCCA/PrACo++), selected counting external dataset(s) with annotation-compatible endpoints; potential second frozen backbone | Target 2+ independent regimes with directional benefit and at least one statistically supported primary endpoint + safety noninferiority | Narrow claims or stop paper |
| **G5: SOTA/venue** | Shared | Official compatible baselines (CountGD++, DAVE, ValidCounter, CountSE, SAM3Count, AdaCount as applicable), compute-matched | Claim SOTA only if actual matched benchmark result exceeds published/reproduced peers; A* gate requires method contribution even if not SOTA | Paper analysis or workshop, no false SOTA claim |

**Dataset discipline:** FSC147 TEST, PrACo Test-1 and V14 are already exposed to the project; cannot call newly integrated method's tuning on them confirmatory. Freeze a new TRAIN/DEV/CAL/FINAL_SEALED split by source image (and by category where target is unseen classes). Check duplicate/near-duplicate image and class leakage. Annotate full and partial instances, confusers, overlaid instances, identity of crop candidates, absent targets. Sample selection rules must be frozen from baseline without using FQIE test outcomes.

**Stats:** clustered paired bootstrap by source-image (not naïve per-query when queries share images), paired MAE and exact accuracy, false-promotion safety with predeclared noninferiority margin, region precision/recall, false merge/split, UNKNOWN and risk–coverage, cost (Qwen calls/tokens, latency, VRAM, tile count). For risk/certificates, state explicit statistical assumptions and verify empirical coverage on CAL and untouched TEST.

## 8. Task split / PR plan — một codebase, không ba method

1. **PR-01 Shared:** freeze JSON source protocol registry for old operating points; create native output hash and parity fixture, no GT inputs.
2. **PR-02 Lộc:** official CountGD++ adapter with pseudo-exemplar rerun; compare to original CountGD++ native predictions exactly (V14 cannot use no-pseudo adapter); preflight integration test on GPU.
3. **PR-03 Lộc:** Qwen region model actual GPU smoke, fixed prompt/model revision, parser failure audit and runtime. Start from `integrations/qwen_unary.py`, not notebook-as-entrypoint.
4. **PR-04 Lộc:** multi-view recovery candidate quality and cost; compare to V7.5/Thúy PR with explicit protocol tags; no raw tile union as method.
5. **PR-05 Thư:** build new-clean training features and candidate ID mapping; `thu_features.py` test with risk router and model calibration, compare to simple margin. Do not use exposed FSC nodes for fresh test.
6. **PR-06 Thúy:** V14 raw integrity + full cohort reconstruction; match summary artifact hashes; document freeze timestamps.
7. **PR-07 Thúy:** candidate graph unifies baseline+rescue, same-view duplicate diagnostics, verified nontransitive identity edges, true region-level GT metrics.
8. **PR-08 Shared:** calibrate uncertainty and policy safety on CAL only; threshold vs dual-NO controlled validation.
9. **PR-09 Thúy+Thư:** design joint arbitration objective/solver; complexity tests, risk/cost ablations, compare simple heuristics.
10. **PR-10 Shared:** source-disjoint external datasets plus strong matched baselines, statistical pipeline.
11. **PR-11 Shared:** reproducible experimental release, figure generation, failure analysis, runtime/VRAM profiling.
12. **PR-12 Shared:** paper manuscript/claim ledger only after G0–G4.

Every PR must include exact input/output schema, protocol ID, no-GT assertion, tests and a small create-only frozen fixture; never modify or force-push historical results. Branch `research/unified-integration-verified` is an **engineering integration branch** to review/PR, not an A* paper release.

## 9. Reproduction command & known blockers

```bash
# from cloned repo
python -m pip install -e '.[test]'
python -m pytest -q tests legacy/thuy/branch/test_*.py legacy/loc/tools/praco_test1/test_verify_praco_artifacts.py

# actual backend readiness (does not download or run GPU)
countlab-preflight --config configs/integrations/native_single_pass_example.json

# CPU replay, already working
countlab-method --input fixtures/unified_toy.json --config configs/method/unicount_v1.json --out toy_result.json

# GPU online (AFTER editing config with real upstream/checkpoint/BERT/Qwen revision)
countlab-online --image /path/to/image.jpg --image-id x --query-id q1 \
  --query 'chair' --config configs/integrations/native_single_pass_example.json \
  --out-dir outputs/run_q1
```

Current local preflight returns **BLOCKED**: missing CountGD++ repo/checkpoint/BERT paths, missing `transformers` and `qwen_vl_utils`, unset Qwen exact revision. This is **correct failure behavior** and no fake 37.5% online gain is reported. Pin official Lộc Qwen V7.6 revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` only for strict reproduction of that branch; this integrated region prompt has a **different** protocol and needs its own frozen version.

## 10. Final decision

**SOFTWARE GO:** one root method and one optional online GPU code path, traceable to each owner's source, CPU tests passed; ready for team review and server integration.

**SCIENTIFIC HOLD:** A* novelty not established, V14 raw/pseudo parity absent, learned Thư transfer not proven, unknown/identity decision quality not benchmarked. The correct near-term target is **G0 + G1**, not public SOTA claim.
