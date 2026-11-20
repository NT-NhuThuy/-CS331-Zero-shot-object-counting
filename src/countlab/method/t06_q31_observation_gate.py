"""Validate one real q31 C2 observation without inventing evidence."""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKET = ROOT / "runs/thuy_t06_verifier_packet_20261118"

REQUESTS_SHA = "15720bb6eba1b94087378061551a0992635b8ecac72cb3a4ec8da90390a71f43"
REQUEST_ID = "t06pilot:q31:pair:d1017:d1018"
CANDIDATE_A = "cgdpp:q31:d1017"
CANDIDATE_B = "cgdpp:q31:d1018"

REQUIRED_OBSERVATION = {
    "request_id",
    "raw_response",
    "parsed_answer",
    "model_revision",
    "prompt_sha256",
    "image_sha256",
    "time_ms",
    "tokens_or_gpu_cost",
    "evidence_ref",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError("STOP: " + message)


def load_packet():
    requests_path = PACKET / "verification_requests.jsonl"
    manifest_path = PACKET / "packet_manifest.json"

    require(requests_path.is_file(), "REQUESTS_MISSING")
    require(manifest_path.is_file(), "PACKET_MANIFEST_MISSING")

    data = requests_path.read_bytes()
    require(sha(data) == REQUESTS_SHA, "REQUESTS_SOURCE_HASH_CHANGED")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(
        manifest.get("requests_sha256") == sha(data),
        "REQUEST_MANIFEST_HASH_MISMATCH",
    )

    records = [
        json.loads(line)
        for line in data.decode("utf-8").splitlines()
        if line.strip()
    ]

    matches = [
        row for row in records
        if row.get("request_id") == REQUEST_ID
        and row.get("op") == "PAIR_IDENTITY"
        and row.get("a") == CANDIDATE_A
        and row.get("b") == CANDIDATE_B
    ]

    require(len(matches) == 1, "EXACT_Q31_PAIR_REQUEST_NOT_UNIQUE")
    request = matches[0]

    require(
        request.get("query_id") == "fsc147:query:31"
        and request.get("query_text") == "ski .",
        "Q31_QUERY_IDENTITY_CHANGED",
    )
    require(
        request.get("status") == "REQUESTED_NOT_OBSERVED",
        "SOURCE_REQUEST_STATUS_CHANGED",
    )
    require(
        set(request.get("allowed_answers", []))
        == {"SAME", "DIFFERENT", "UNKNOWN"},
        "ALLOWED_ANSWERS_CHANGED",
    )

    review = PACKET / request["review_image"]
    require(review.is_file(), "REVIEW_IMAGE_MISSING")
    require(
        sha(review.read_bytes()) == request["review_image_sha256"],
        "REVIEW_IMAGE_HASH_MISMATCH",
    )

    return manifest, request, sha(manifest_path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-image", type=Path)
    parser.add_argument("--observation", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    packet_manifest, request, manifest_sha = load_packet()

    image_path = (
        args.source_image
        if args.source_image is not None
        else Path(request["current_image_path"])
    )

    source_verified = (
        image_path.is_file()
        and sha(image_path.read_bytes()) == request["image_sha256"]
    )

    print("Q31_REQUEST_ID:", REQUEST_ID)
    print("QUERY_ID:", request["query_id"])
    print("CANDIDATE_A:", CANDIDATE_A)
    print("CANDIDATE_B:", CANDIDATE_B)
    print("SOURCE_IMAGE_PATH:", image_path)
    print("SOURCE_IMAGE_SHA256_VERIFIED:", source_verified)
    print("REVIEW_IMAGE_SHA256_VERIFIED: True")
    print(
        "HISTORICAL_LOADER_ATTESTED:",
        request.get("historical_loader_attested"),
    )
    print(
        "PACKET_OBSERVED_EVENTS:",
        packet_manifest.get("observed_verifier_events"),
    )

    if args.observation is None:
        require(args.out_dir is None, "OUT_DIR_WITHOUT_OBSERVATION")
        print(
            "STATUS:",
            "READY_TO_VALIDATE_C2_OBSERVATION"
            if source_verified
            else "HOLD_SOURCE_IMAGE_NOT_LOCALLY_VERIFIED",
        )
        print("NEW_MODEL_CALLS: 0")
        print("FILES_MODIFIED: False")
        return

    require(args.out_dir is not None, "OUT_DIR_REQUIRED")
    require(source_verified, "SOURCE_IMAGE_HASH_NOT_VERIFIED")
    require(args.observation.is_file(), "C2_OBSERVATION_FILE_MISSING")
    require(not args.out_dir.exists(), "OUTPUT_DIRECTORY_ALREADY_EXISTS")

    raw_bytes = args.observation.read_bytes()
    observation = json.loads(raw_bytes)

    require(isinstance(observation, dict), "OBSERVATION_NOT_OBJECT")
    require(
        REQUIRED_OBSERVATION.issubset(observation),
        "MISSING_REQUIRED_C2_OBSERVATION_FIELDS",
    )
    require(
        observation["request_id"] == REQUEST_ID,
        "OBSERVATION_REQUEST_ID_MISMATCH",
    )
    require(
        observation["image_sha256"] == request["image_sha256"],
        "OBSERVATION_IMAGE_HASH_MISMATCH",
    )
    require(
        observation["parsed_answer"] in request["allowed_answers"],
        "INVALID_OBSERVED_IDENTITY_VERDICT",
    )
    require(
        isinstance(observation["raw_response"], (str, dict))
        and bool(observation["raw_response"]),
        "RAW_QWEN_RESPONSE_MISSING",
    )
    require(
        isinstance(observation["model_revision"], str)
        and bool(observation["model_revision"].strip()),
        "MODEL_REVISION_MISSING",
    )
    require(
        isinstance(observation["prompt_sha256"], str)
        and re.fullmatch(
            r"[0-9a-fA-F]{64}",
            observation["prompt_sha256"],
        ) is not None,
        "PROMPT_HASH_INVALID",
    )
    require(
        isinstance(observation["evidence_ref"], str)
        and bool(observation["evidence_ref"].strip()),
        "EVIDENCE_REF_MISSING",
    )

    latency = observation["time_ms"]
    require(
        isinstance(latency, (int, float))
        and not isinstance(latency, bool)
        and math.isfinite(latency)
        and latency > 0,
        "INVALID_MEASURED_LATENCY",
    )

    cost = observation["tokens_or_gpu_cost"]
    require(
        (isinstance(cost, dict) and bool(cost))
        or (
            isinstance(cost, (int, float))
            and not isinstance(cost, bool)
            and math.isfinite(cost)
            and cost > 0
        ),
        "MEASURED_TOKENS_OR_COST_MISSING",
    )

    forbidden = {
        "gt", "gt_count", "ground_truth", "true_count",
        "gold_count", "count_label", "gt_annotations",
    }
    require(
        not (forbidden & {str(key).lower() for key in observation}),
        "GT_FIELDS_FOUND_IN_C2_OBSERVATION",
    )

    event = {
        "op": "PAIR_IDENTITY",
        "a": request["a"],
        "b": request["b"],
        "relation": observation["parsed_answer"],
        "evidence_ref": observation["evidence_ref"],
    }

    from countlab.method.joint_solver_v2 import validate_event_contract
    validate_event_contract([event])

    execution_log = {
        "schema": "c2_c3_q31_real_observation_log_v1",
        "request_id": REQUEST_ID,
        "query_id": request["query_id"],
        "request_file_sha256": REQUESTS_SHA,
        "packet_manifest_sha256": manifest_sha,
        "observation_file_sha256": sha(raw_bytes),
        "source_image_sha256": request["image_sha256"],
        "source_image_hash_locally_verified": True,
        "historical_loader_attested": False,
        "event_sha256": sha(
            json.dumps(event, sort_keys=True).encode("utf-8")
        ),
        "observation": observation,
        "full_query_pool_verified": False,
        "accuracy_evaluation": False,
    }

    parent = args.out_dir.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".q31_gate_", dir=parent))

    try:
        (staging / "observed_events.jsonl").write_text(
            json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "execution_log.json").write_text(
            json.dumps(execution_log, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.copyfile(
            args.observation,
            staging / "original_c2_observation.json",
        )
        require(
            not args.out_dir.exists(),
            "OUTPUT_DIRECTORY_CREATED_CONCURRENTLY",
        )
        os.rename(staging, args.out_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    print("T06_Q31_OBSERVATION_GATE: PASS")
    print("PARSED_VERDICT:", event["relation"])
    print("EVIDENCE_REF:", event["evidence_ref"])
    print("OUTPUT:", args.out_dir)
    print("FULL_QUERY_POOL_VERIFIED: False")
    print("MAE_COMPUTED: False")
    print("NEW_MODEL_CALLS: 0")


if __name__ == "__main__":
    main()
