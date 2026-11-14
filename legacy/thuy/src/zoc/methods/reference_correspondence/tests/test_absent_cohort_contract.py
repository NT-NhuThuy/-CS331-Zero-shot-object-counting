import csv
import tempfile
import unittest
from pathlib import Path

from zoc.methods.reference_correspondence.evaluate import load_labels, validate_cohort_ids


FIELDS = ["pair_id", "presence_label", "count_target", "final_decision", "benchmark_inclusion"]


def write_manifest(path: Path, ids: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for pair_id in ids:
            writer.writerow({"pair_id": pair_id, "presence_label": 0, "count_target": 0,
                             "final_decision": "KEEP_NEGATIVE", "benchmark_inclusion": "YES"})


class AbsentCohortContractTests(unittest.TestCase):
    def test_exact_111_ids_are_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "absent.csv"
            write_manifest(path, [f"p{i:03d}" for i in range(111)])
            labels = load_labels("fsc_verified_absent111", absent_manifest=path)
            self.assertEqual(len(labels), 111)
            validate_cohort_ids([{"item_id": key} for key in labels], labels)

    def test_subset_is_rejected_by_cohort_id_check(self):
        labels = {f"fsc_verified_absent:p{i:03d}": {} for i in range(111)}
        with self.assertRaisesRegex(ValueError, "label/input cohort mismatch"):
            validate_cohort_ids([{"item_id": "fsc_verified_absent:p000"}], labels)

    def test_duplicate_pair_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "absent.csv"
            write_manifest(path, [f"p{i:03d}" for i in range(110)] + ["p000"])
            with self.assertRaisesRegex(ValueError, "duplicate pair_id"):
                load_labels("fsc_verified_absent111", absent_manifest=path)

    def test_wrong_identity_is_rejected(self):
        labels = {f"fsc_verified_absent:p{i:03d}": {} for i in range(111)}
        rows = [{"item_id": f"fsc_verified_absent:p{i:03d}"} for i in range(110)]
        rows.append({"item_id": "fsc_verified_absent:wrong"})
        with self.assertRaisesRegex(ValueError, "label/input cohort mismatch"):
            validate_cohort_ids(rows, labels)
