from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from .io import read_jsonl, read_json, write_json, file_hash
from .protocol import validate_row
from .infer import selected_points


def point_match(predicted, ground_truth, radius):
    """Maximum valid match cardinality, then minimum distance. Not greedy matching.

    Plain min-distance Hungarian followed by radius filtering can lose valid TPs.
    Invalid edges have a cost larger than the TOTAL cost of all valid edges.
    """
    if radius <= 0 or not math.isfinite(radius):
        raise ValueError('Radius must be finite and positive')
    p = np.asarray(predicted, dtype=float).reshape(-1, 2)
    g = np.asarray(ground_truth, dtype=float).reshape(-1, 2)
    if not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError('Nonfinite point coordinates')
    if not len(p) or not len(g):
        return 0, len(p), len(g)
    distance = np.linalg.norm(p[:, None, :] - g[None, :, :], axis=-1)
    valid = distance <= radius
    invalid_cost = min(len(p), len(g)) + 1.0
    cost = np.where(valid, distance / radius, invalid_cost)
    rows, cols = linear_sum_assignment(cost)
    tp = int(valid[rows, cols].sum())
    return tp, len(p) - tp, len(g) - tp


def summarize(per_query):
    known = [r for r in per_query if r['status'] != 'unknown']
    positive = [r for r in known if r['status'] == 'present']
    absent = [r for r in known if r['status'] == 'absent']
    tp, fp, fn = (sum(r[k] for r in known) for k in ('tp', 'fp', 'fn'))
    by_class = defaultdict(list)
    for r in positive:
        by_class[r['query']].append(abs(r['pred_count'] - r['gt_count']))
    errors = [r['pred_count'] - r['gt_count'] for r in positive]
    return {'queries': len(per_query), 'present': len(positive), 'verified_absent': len(absent),
            'unknown_excluded_from_accuracy': len(per_query) - len(known),
            'positive_mae': float(np.mean(np.abs(errors))) if errors else None,
            'positive_rmse': float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            'positive_class_macro_mae': float(np.mean([np.mean(v) for v in by_class.values()])) if by_class else None,
            'absent_mae': float(np.mean([r['pred_count'] for r in absent])) if absent else None,
            'absent_false_count_rate': float(np.mean([r['pred_count'] > 0 for r in absent])) if absent else None,
            'point_micro_precision': tp / (tp + fp) if tp + fp else None,
            'point_micro_recall': tp / (tp + fn) if tp + fn else None,
            'point_micro_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            'positive_point_macro_f1': float(np.mean([r['f1'] for r in positive])) if positive else None,
            'tp': tp, 'fp': fp, 'fn': fn}


def evaluate(ds, manifest, predictions_dir, output, baseline_threshold=None,
             pool_threshold=None, grounding_threshold=None, allow_partial=False):
    predictions_dir, output = Path(predictions_dir), Path(output)
    run = read_json(predictions_dir / 'run.json')
    if run['dataset_fingerprint'] != ds.fingerprint:
        raise ValueError('Prediction/dataset mismatch')
    thresholds = dict(run['thresholds_at_inference'])
    for k, v in (('baseline_threshold', baseline_threshold), ('pool_threshold', pool_threshold),
                 ('grounding_threshold', grounding_threshold)):
        if v is not None:
            if not 0 <= v <= 1:
                raise ValueError('Thresholds must be in [0,1]')
            thresholds[k] = v
    rows = list(read_jsonl(manifest))
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate manifest IDs')
    modes = ['baseline'] + (['verifier'] if run['verifier_sha256'] else [])
    result_rows, missing = {m: [] for m in modes}, []
    image_hashes = {}
    for row in rows:
        validate_row(ds, row)
        path = predictions_dir / 'items' / (row['id'] + '.json')
        if not path.exists():
            missing.append(row['id'])
            continue
        item = read_json(path)
        if row['image'] not in image_hashes:
            image_hashes[row['image']] = file_hash(ds.image_path(row['image']))
        if item['image_sha256'] != image_hashes[row['image']]:
            raise ValueError('Image bytes changed after inference; recompute predictions')
        for key in ('id', 'image', 'query', 'split', 'width', 'height', 'dataset_fingerprint'):
            if item[key] != row[key]:
                raise ValueError(f'Prediction identity mismatch: {key}')
        if item['run_id'] != run['id']:
            raise ValueError('Predictions mix different runs')
        radius = ds.cfg['point_radius_relative'] * math.hypot(row['width'], row['height'])
        for mode in modes:
            points, _ = selected_points(item, mode, **thresholds)
            if row['status'] == 'unknown':
                tp = fp = fn = f1 = None
            else:
                tp, fp, fn = point_match(points, row['gt_points'], radius)
                f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
            result_rows[mode].append({'id': row['id'], 'image': row['image'], 'query': row['query'],
                'split': row['split'], 'status': row['status'], 'gt_count': row['gt_count'],
                'pred_count': len(points), 'tp': tp, 'fp': fp, 'fn': fn, 'f1': f1,
                'radius_pixels': radius})
    if missing and not allow_partial:
        raise ValueError(f'{len(missing)}/{len(rows)} predictions missing. Explicitly use --allow-partial for a smoke run.')
    report = {'schema': 1, 'run_id': run['id'], 'thresholds': thresholds,
              'point_matching': 'max-cardinality one-to-one; Euclidean radius = relative * image diagonal',
              'point_radius_relative': ds.cfg['point_radius_relative'],
              'requested_queries': len(rows), 'missing_queries': len(missing), 'partial': bool(missing),
              'metrics': {m: summarize(values) for m, values in result_rows.items()}}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'metrics.json', report)
    fields = ['id', 'image', 'query', 'split', 'status', 'gt_count', 'pred_count', 'tp', 'fp', 'fn', 'f1', 'radius_pixels']
    for mode, values in result_rows.items():
        with open(output / (mode + '.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    return report
