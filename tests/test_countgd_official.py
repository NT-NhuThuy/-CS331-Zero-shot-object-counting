"""CPU software contracts only. No weights, GPU parity, or model-accuracy claims."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from countlab.integrations import countgd_official as official
from countlab.integrations.countgd_native import NativeCountGD, PROTOCOL_ID as NATIVE_ID
from countlab.integrations.online import execute_online
from countlab.integrations.preflight import inspect
from countlab.method.schema import parse_input

ROOT = Path(__file__).resolve().parents[1]


def config():
    cfg = json.loads((ROOT / "configs/integrations/official_pseudo3_example.json").read_text())
    p = cfg["official_countgd"]["protocol"]
    p["checkpoint_sha256"] = "a" * 64
    p["cuda_extension_sha256"] = "b" * 64
    p["bert_files_sha256"] = {name: "c" * 64 for name in p["bert_files_sha256"]}
    p["library_versions"] = {name: "test-1.0" for name in official.LIBRARIES}
    p["cuda_runtime"] = "test-cuda"
    cfg["official_countgd"]["protocol_sha256"] = official.canonical_sha(p)
    return cfg


class FixtureEngine:
    """Recorded synthetic boundary between orchestration and torch engine."""
    def __init__(self):
        self.fixture = json.loads((ROOT / "fixtures/countgd_official_software.json").read_text())
        self.resized_size = tuple(self.fixture["resized_size"])
        self.calls = []
    def prepare(self, path):
        self.calls = []
    def forward(self, caption, exemplars):
        self.calls.append((caption, copy.deepcopy(exemplars)))
        return copy.deepcopy(self.fixture["first" if len(self.calls) == 1 else "second"])
    def select_pseudo(self, raw, *, num_exemplars):
        assert num_exemplars == 3
        return self.fixture["selected_pseudo_resized_xyxy"]


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "synthetic.png"
    Image.new("RGB", (100, 80), color=(31, 47, 59)).save(path)
    return path


def test_two_forwards_preserve_schema_geometry_and_provenance(image):
    cfg = config(); engine = FixtureEngine()
    provider = official.OfficialCountGD(cfg["official_countgd"], engine=engine)
    manifest, result = execute_online(image_path=image, query="chair", image_id="synthetic",
        query_id="p02", config=cfg, backend=provider)
    assert engine.calls == [("chair .", []), ("chair .", engine.fixture["selected_pseudo_resized_xyxy"])]
    assert len(manifest["baseline"]) == result["count"] == 2
    assert manifest["baseline"][0]["box_xyxy"] == [30, 20, 70, 60]
    assert manifest["baseline"][1]["box_xyxy"] == [0, 20, 20, 60]
    parse_input(manifest)
    trace = manifest["metadata"]["detector_runs"][0]
    assert trace["detector_calls"] == 2 and trace["initial_exemplars"] == 0
    assert trace["original_size"] == [100, 80] and trace["resized_size"] == [200, 160]
    assert trace["peak_vram_bytes"] is None
    assert trace["caption_sha256"] == hashlib.sha256(b"chair .").hexdigest()
    assert provider.provenance["evidence_level"] == "SOFTWARE_CONTRACT_TEST"
    assert provider.provenance["asset_validation"] == "NOT_CHECKED_TEST_DOUBLE"
    assert result["evidence_trace"][-1]["stage"] == "software_contract_test"
    assert manifest["metadata"]["backend_protocol"] == official.PROTOCOL_ID
    assert any("HOLD for V14 parity" in s for s in result["limitations"])
    assert not any("uses no pseudo" in s for s in result["limitations"])


def test_ids_are_stable_and_namespaced_by_image_query_role_view_and_offset(image):
    provider = official.OfficialCountGD(config()["official_countgd"], engine=FixtureEngine())
    def infer(query="chair", **kwargs):
        return provider.infer(image, query, image_id="im", **kwargs)
    a = infer(); b = infer()
    assert [p["candidate_id"] for p in a] == [p["candidate_id"] for p in b]
    assert len({p["candidate_id"] for p in a}) == 2
    originals = {p["candidate_id"] for p in a}
    for values in [infer("table"), infer(role="rescue"), infer(view_id="tile"), infer(offset=(50, 10))]:
        assert not originals & {p["candidate_id"] for p in values}
    changed = infer(offset=(50, 10))
    assert changed[0]["box_xyxy"] == [80, 30, 120, 70]


def test_empty_first_pass_still_reruns_the_detector(image):
    engine = FixtureEngine()
    engine.fixture["first"]["boxes_cxcywh_normalized"] = []
    engine.fixture["first"]["token_probabilities"] = []
    engine.fixture["selected_pseudo_resized_xyxy"] = []
    provider = official.OfficialCountGD(config()["official_countgd"], engine=engine)
    assert len(provider.infer(image, "chair")) == 2
    assert len(engine.calls) == 2 and engine.calls[1][1] == []


def test_forward_failure_is_not_replaced_by_first_stage_counts(image):
    class FailingEngine(FixtureEngine):
        def forward(self, caption, exemplars):
            if self.calls:
                raise RuntimeError("second forward failed")
            return super().forward(caption, exemplars)
    provider = official.OfficialCountGD(config()["official_countgd"], engine=FailingEngine())
    with pytest.raises(RuntimeError, match="second forward failed"):
        provider.infer(image, "chair")
    assert provider.runs == []


def test_invalid_pseudo_coordinates_fail_before_second_forward(image):
    engine = FixtureEngine()
    engine.fixture["selected_pseudo_resized_xyxy"] = [[0, 0, 300, 160]]
    provider = official.OfficialCountGD(config()["official_countgd"], engine=engine)
    with pytest.raises(ValueError, match="pseudo-exemplar"):
        provider.infer(image, "chair")
    assert len(engine.calls) == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_nonfinite_or_invalid_scores_are_not_hidden_by_filtering(value):
    raw = FixtureEngine().fixture["second"]
    raw["token_probabilities"][-1][0] = value
    with pytest.raises(ValueError, match="token probabilities"):
        official.candidates_from_prediction(raw, width=100, height=80, namespace="test", role="baseline", view_id="full")


def test_content_filter_threshold_and_raw_query_index_are_preserved():
    raw = FixtureEngine().fixture["second"]
    raw["token_probabilities"][0] = [0.1, 0.23, 0.1, 0.1]
    raw["token_probabilities"][1] = [0.9, 0.0, 0.1, 0.1]
    assert official.candidates_from_prediction(raw, width=100, height=80, namespace="test", role="baseline", view_id="full") == []


@pytest.mark.parametrize("box", [[2, 2, .1, .1], [.5, .5, 0, .1], [float("nan"), .5, .1, .1]])
def test_nonphysical_boxes_cannot_be_fabricated_or_silently_counted(box):
    raw = FixtureEngine().fixture["second"]
    raw["boxes_cxcywh_normalized"][0] = box
    with pytest.raises(ValueError, match="box"):
        official.candidates_from_prediction(raw, width=100, height=80, namespace="test", role="baseline", view_id="full")


@pytest.mark.parametrize("offset", [(-1, 0), (0, float("nan"))])
def test_invalid_coordinate_offsets_fail(offset):
    with pytest.raises(ValueError, match="offset"):
        official.candidates_from_prediction(FixtureEngine().fixture["second"], width=100, height=80,
            namespace="test", role="rescue", view_id="tile", offset=offset)


def test_density_count_only_backend_is_outside_candidate_schema():
    with pytest.raises(ValueError, match="count-only"):
        official.candidates_from_prediction({"count": 3, "density": []}, width=100, height=80,
            namespace="test", role="baseline", view_id="full")


@pytest.mark.parametrize("field,value", [("num_pseudo_exemplars", 0), ("num_exemplars", 1),
    ("box_threshold", .1), ("precision", "float16"), ("seed", 123), ("crop", True)])
def test_frozen_operating_point_cannot_be_changed_even_with_rehashed_config(field, value):
    cfg = config()["official_countgd"]
    cfg["protocol"]["operating_point"][field] = value
    cfg["protocol_sha256"] = official.canonical_sha(cfg["protocol"])
    with pytest.raises(ValueError, match="operating point"):
        official.OfficialCountGD(cfg, engine=FixtureEngine())


def test_protocol_digest_mismatch_rejected():
    cfg = config()["official_countgd"]; cfg["protocol_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="protocol_sha256 mismatch"):
        official.OfficialCountGD(cfg, engine=FixtureEngine())


def test_missing_assets_fail_before_torch_or_model_loading():
    cfg = config()["official_countgd"]
    with pytest.raises(RuntimeError, match="assets BLOCKED"):
        official.OfficialCountGD(cfg)
    result = inspect(json.loads((ROOT / "configs/integrations/official_pseudo3_example.json").read_text()))
    assert result["status"] == "BLOCKED" and result["parity_status"] == "HOLD for V14 parity"
    assert any("checkpoint_sha256" in b for b in result["blockers"])


@pytest.mark.parametrize("where", ["config", "raw"])
def test_gt_fields_fail_before_becoming_inference_evidence(image, where):
    cfg = config(); engine = FixtureEngine()
    if where == "raw": engine.fixture["second"]["gt_count"] = 42
    else: cfg["official_countgd"]["protocol"]["gt_count"] = 42
    with pytest.raises(ValueError, match="GT/label"):
        provider = official.OfficialCountGD(cfg["official_countgd"], engine=engine)
        provider.infer(image, "chair")


def test_official_selection_cannot_relabel_a_no_pseudo_provider(image):
    class WrongProvider:
        protocol_id = NATIVE_ID
    with pytest.raises(ValueError, match="backend does not match"):
        execute_online(image_path=image, query="chair", image_id="im", query_id="q",
                       config=config(), backend=WrongProvider())


def test_runner_rejects_provider_with_different_checkpoint_protocol_hash(image):
    requested = config(); loaded = config()
    loaded["official_countgd"]["protocol"]["checkpoint_sha256"] = "d" * 64
    loaded["official_countgd"]["protocol_sha256"] = official.canonical_sha(loaded["official_countgd"]["protocol"])
    engine = FixtureEngine()
    provider = official.OfficialCountGD(loaded["official_countgd"], engine=engine)
    with pytest.raises(ValueError, match="protocol hash differs"):
        execute_online(image_path=image, query="chair", image_id="im", query_id="q",
                       config=requested, backend=provider)
    assert engine.calls == []


def test_runner_rejects_gt_before_detector_calls(image):
    cfg = config(); cfg["metadata"] = {"gt_count": 42}
    engine = FixtureEngine()
    provider = official.OfficialCountGD(cfg["official_countgd"], engine=engine)
    with pytest.raises(ValueError, match="GT/label"):
        execute_online(image_path=image, query="chair", image_id="im", query_id="q", config=cfg, backend=provider)
    assert engine.calls == []


def test_missing_hashes_and_malformed_config_cannot_be_ready():
    assert official.inspect_assets([])["status"] == "BLOCKED"
    cfg = config()["official_countgd"]
    cfg["protocol"]["bert_files_sha256"]["../vocab.txt"] = "0" * 64
    cfg["protocol_sha256"] = official.canonical_sha(cfg["protocol"])
    with pytest.raises(ValueError, match="BERT asset"):
        official.validate_protocol(cfg)


def test_no_pseudo_native_regression_remains_one_call_and_same_ids(image):
    class NativeEngine:
        calls = 0
        def predict(self, path, query):
            self.calls += 1
            return {"boxes_cxcywh_normalized": [[.5, .5, .4, .5]], "native_scores": [.9],
                    "native_text_pass": [True], "original_size": [100, 80]}
    engine = NativeEngine(); provider = NativeCountGD({"threshold": .23}, backend=engine)
    candidates = provider.infer(image, "chair", image_id="im")
    assert engine.calls == 1 and provider.protocol_id == NATIVE_ID
    assert candidates[0]["candidate_id"] == "baseline:full:0"
    assert candidates[0]["box_xyxy"] == [30, 20, 70, 60]


def test_real_gpu_smoke_fixture_is_gt_blind_and_keeps_its_recorded_hash():
    """Validate a recorded real smoke artifact; this test does not rerun GPU."""
    raw = (ROOT / "fixtures/countgd_official_gpu_smoke.json").read_bytes()
    summary = json.loads((ROOT / "fixtures/countgd_official_gpu_smoke_summary.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == summary["candidate_fixture_sha256"]
    data = json.loads(raw)
    baseline, rescue, edges = parse_input(data)
    assert len(baseline) == summary["detector_run"]["output_candidates"] == 16
    assert not rescue and not edges
    assert summary["gpu_smoke"] == "PASS"
    assert summary["v14_count_parity"] == summary["v14_bbox_parity"] == "NOT MEASURED"
    width, height = summary["detector_run"]["original_size"]
    assert all(0 <= p.box[0] < p.box[2] <= width and 0 <= p.box[1] < p.box[3] <= height for p in baseline)
    assert b"/home/" not in raw and b"C:\\Users\\" not in raw


@pytest.fixture
def local_asset_lock(tmp_path, monkeypatch):
    """Tiny synthetic assets exercise hash checks, not a real model loader."""
    repo = tmp_path / "upstream"; repo.mkdir()
    blobs = {}
    for name in official.SOURCE_BLOBS:
        path = repo / name; path.parent.mkdir(parents=True, exist_ok=True)
        data = b"synthetic source fixture\n"; path.write_bytes(data)
        blobs[name] = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    monkeypatch.setattr(official, "SOURCE_BLOBS", blobs)
    cfg = config()["official_countgd"]
    cfg["protocol"]["source_git_blobs"] = blobs
    checkpoint = tmp_path / "checkpoint.bin"; checkpoint.write_bytes(b"NOT MODEL WEIGHTS")
    cfg["repo"] = str(repo); cfg["checkpoint"] = str(checkpoint)
    cfg["protocol"]["checkpoint_sha256"] = official.file_sha(checkpoint)
    bert = tmp_path / "bert"; bert.mkdir()
    for name in cfg["protocol"]["bert_files_sha256"]:
        path = bert / name; path.write_bytes(b"synthetic bert fixture")
        cfg["protocol"]["bert_files_sha256"][name] = official.file_sha(path)
    cfg["bert_dir"] = str(bert)
    extension = tmp_path / "cuda.so"; extension.write_bytes(b"NOT A CUDA BINARY")
    cfg["protocol"]["cuda_extension_sha256"] = official.file_sha(extension)
    cfg["protocol_sha256"] = official.canonical_sha(cfg["protocol"])
    monkeypatch.setattr(official, "_git", lambda repo, *args: official.SOURCE_COMMIT if args[0] == "rev-parse" else "")
    monkeypatch.setattr(official.importlib.metadata, "version", lambda name: "test-1.0")
    monkeypatch.setattr(official.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(extension)))
    return cfg


def test_all_synthetic_asset_hashes_pass_preflight_without_loading(local_asset_lock):
    result = official.inspect_assets(local_asset_lock)
    assert result["status"] == "READY_FOR_GPU_SMOKE_NOT_MODEL_PARITY"
    assert result["parity_status"] == "HOLD for V14 parity"


@pytest.mark.parametrize("asset", ["checkpoint", "bert", "source", "cuda", "commit", "dirty"])
def test_asset_or_source_mismatch_blocks_loading(local_asset_lock, monkeypatch, asset):
    cfg = local_asset_lock
    if asset == "checkpoint": Path(cfg["checkpoint"]).write_bytes(b"changed")
    elif asset == "bert": (Path(cfg["bert_dir"]) / "vocab.txt").write_bytes(b"changed")
    elif asset == "source": (Path(cfg["repo"]) / "cfg_app.py").write_bytes(b"changed")
    elif asset == "cuda":
        cfg["protocol"]["cuda_extension_sha256"] = "0" * 64
        cfg["protocol_sha256"] = official.canonical_sha(cfg["protocol"])
    elif asset == "commit": monkeypatch.setattr(official, "_git", lambda *args: "wrong")
    else: monkeypatch.setattr(official, "_git", lambda repo, *args: official.SOURCE_COMMIT if args[0] == "rev-parse" else " M cfg_app.py")
    assert official.inspect_assets(cfg)["status"] == "BLOCKED"
