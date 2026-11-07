from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from .model import RegionTextVerifier
from .geometry import cxcywh_to_xyxy
from .io import load_tensor_file, read_jsonl, write_json, read_json, digest, file_hash
from .protocol import validate_row


def score_proposals(backend, model, prediction, query, chunk_size=64):
    boxes = cxcywh_to_xyxy(prediction['boxes_cxcywh_normalized']).clamp(0, 1)
    usable = ((boxes[:, 2:] - boxes[:, :2]) > 0).all(dim=1)
    indices = torch.where(usable)[0]
    # Invalid/degenerate boxes are not silently sent to ROIAlign.
    scores = torch.zeros(len(boxes), device=boxes.device)
    text = backend.encode_text(query)
    with torch.no_grad():
        for start in range(0, len(indices), chunk_size):
            selected = indices[start:start + chunk_size]
            visual = backend.proposal_tokens(prediction, selected)
            text_batch = text.unsqueeze(0).expand(len(selected), -1, -1)
            valid = torch.ones(text_batch.shape[:2], dtype=torch.bool, device=text.device)
            values = model(visual, text_batch, valid).sigmoid()
            if not torch.isfinite(values).all():
                raise RuntimeError('Nonfinite verifier output')
            scores[selected] = values
    return scores, usable


def predict_manifest(ds, backend, manifest, output_dir, verifier=None, limit=None, check_hooks=False):
    rows = list(read_jsonl(manifest))
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate manifest IDs')
    if limit:
        rows = rows[:limit]
    model, verifier_sha = None, None
    if verifier:
        saved = load_tensor_file(verifier)
        if saved['dataset_fingerprint'] != ds.fingerprint or saved['backend']['id'] != backend.spec['id']:
            raise ValueError('Verifier dataset/backend differs from extraction setup')
        if set(saved['train_classes']) != ds.class_sets['train']:
            raise ValueError('Verifier train class provenance differs from FSC train split')
        if set(saved['train_classes']) & (ds.class_sets['val'] | ds.class_sets['test']):
            raise ValueError('Verifier has held-out class supervision')
        model = RegionTextVerifier(**saved['settings']).to(backend.device).eval()
        model.load_state_dict(saved['state_dict'], strict=True)
        verifier_sha = file_hash(verifier)
    run = {'schema': 1, 'backend': backend.spec, 'verifier_sha256': verifier_sha,
           'dataset_fingerprint': ds.fingerprint,
           'thresholds_at_inference': {k: ds.cfg[k] for k in
                                      ('baseline_threshold', 'pool_threshold', 'grounding_threshold')}}
    run['id'] = digest(run)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_path = out / 'run.json'
    if run_path.exists() and read_json(run_path)['id'] != run['id']:
        raise ValueError('Output directory belongs to a different run; choose a new directory')
    write_json(run_path, run)
    completed, reused = 0, 0
    checked = False
    for i, row in enumerate(rows, 1):
        validate_row(ds, row)
        if model is not None and row['split'] in ('val', 'test') and row['query'] in saved['train_classes']:
            raise ValueError('Held-out inference query appeared in training')
        image_sha = file_hash(ds.image_path(row['image']))
        path = out / 'items' / (row['id'] + '.json')
        if path.exists():
            old = read_json(path)
            if old['run_id'] != run['id'] or old['image_sha256'] != image_sha:
                raise ValueError('Existing prediction has different provenance; use a new directory')
            reused += 1
            continue
        prediction = backend.predict(ds.image_path(row['image']), row['query'],
                                     check_hooks=check_hooks and not checked)
        checked = checked or check_hooks
        ground, usable = (None, None) if model is None else score_proposals(
            backend, model, prediction, row['query'], ds.cfg['roi_chunk'])
        boxes = prediction['boxes_cxcywh_normalized'].cpu().numpy()
        native = prediction['native_scores'].cpu().numpy()
        w, h = prediction['original_size']
        item = {'schema': 1, 'id': row['id'], 'run_id': run['id'], 'image': row['image'],
                'query': row['query'], 'split': row['split'], 'dataset_fingerprint': ds.fingerprint,
                'image_sha256': image_sha, 'width': w, 'height': h,
                'boxes_cxcywh_normalized': boxes.tolist(), 'native_scores': native.tolist(),
                'native_text_pass': prediction.get('native_text_pass', torch.ones(len(boxes), dtype=torch.bool)).cpu().tolist(),
                'grounding_scores': ground.cpu().tolist() if ground is not None else None,
                'usable_rois': usable.cpu().tolist() if usable is not None else None}
        write_json(path, item)
        completed += 1
        if i % 10 == 0 or i == len(rows):
            print(f'inference {i}/{len(rows)} queries; newly computed={completed}, reused={reused}', flush=True)
    return {'new': completed, 'reused': reused, 'requested': len(rows), 'output': str(out)}


def selected_points(item, mode, baseline_threshold=0.23, pool_threshold=0.23, grounding_threshold=0.5):
    boxes = np.asarray(item['boxes_cxcywh_normalized'], dtype=float).reshape(-1, 4)
    native = np.asarray(item['native_scores'], dtype=float)
    text_pass = np.asarray(item.get('native_text_pass', [True] * len(native)), dtype=bool)
    if text_pass.shape != native.shape:
        raise ValueError('Native text mask shape mismatch')
    if len(boxes) != len(native) or not np.isfinite(boxes).all() or not np.isfinite(native).all():
        raise ValueError('Invalid native prediction')
    if (boxes < 0).any() or (boxes > 1).any() or (native < 0).any() or (native > 1).any():
        raise ValueError('Expected normalized boxes and probability scores')
    if mode == 'baseline':
        keep = (native > baseline_threshold) & text_pass
    elif mode == 'verifier':
        if item['grounding_scores'] is None or item['usable_rois'] is None:
            raise ValueError('No verifier scores in this prediction')
        ground = np.asarray(item['grounding_scores'], dtype=float)
        usable = np.asarray(item['usable_rois'], dtype=bool)
        if ground.shape != native.shape or usable.shape != native.shape or not np.isfinite(ground).all():
            raise ValueError('Invalid verifier scores/ROI mask')
        if (ground < 0).any() or (ground > 1).any():
            raise ValueError('Verifier scores outside [0,1]')
        keep = (native > pool_threshold) & (ground > grounding_threshold) & usable & text_pass
    else:
        raise ValueError(mode)
    points = boxes[keep, :2] * [item['width'], item['height']]
    return points, np.where(keep)[0]
