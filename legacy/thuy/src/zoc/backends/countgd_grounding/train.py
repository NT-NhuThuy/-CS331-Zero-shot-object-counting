from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import torch
from .cache import RegionDataset, collate_regions
from .model import RegionTextVerifier, grounding_loss
from .io import atomic_torch_save, load_tensor_file, write_json


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_training_caches(train, val):
    a, b = train.index, val.index
    if a['split'] != 'train' or b['split'] != 'val':
        raise ValueError('Optimizer uses train cache only; validation must use val cache')
    if a['backend']['id'] != b['backend']['id'] or a['dataset_fingerprint'] != b['dataset_fingerprint']:
        raise ValueError('Train/val backend or dataset mismatch')
    if set(a['allowed_classes']) & set(b['allowed_classes']):
        raise ValueError('Train/val class leakage')
    if {x['image'] for x in a['entries']} & {x['image'] for x in b['entries']}:
        raise ValueError('Train/val image leakage')
    for split, data in (('train', train), ('val', val)):
        counts = {v: sum(x['label'] == v for row in data.entries for x in row['labels']) for v in (0, 1)}
        if not all(counts.values()):
            raise ValueError(f'{split} needs verified positive AND negative pairs; got {counts}. Review null labels first.')


def epoch_pass(model, loader, device, optimizer=None, margin=1.0, cf_weight=0.5):
    model.train(optimizer is not None)
    logits_all, labels_all, total, n_batches = [], [], 0.0, 0
    context = torch.enable_grad if optimizer is not None else torch.no_grad
    with context():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch['visual'], batch['text'], batch['text_valid'])
            loss, _ = grounding_loss(logits, batch['labels'], batch['region_ids'], margin, cf_weight)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; aborting')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            total += loss.item()
            n_batches += 1
            logits_all.extend(logits.detach().cpu().tolist())
            labels_all.extend(batch['labels'].cpu().tolist())
    from scipy.stats import rankdata
    from scipy.special import expit
    labels, logits = np.asarray(labels_all), np.asarray(logits_all)
    probabilities = expit(logits)
    npos, nneg = int((labels == 1).sum()), int((labels == 0).sum())
    auc = float((rankdata(logits)[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else None
    predicted = probabilities >= 0.5
    tp = int((predicted & (labels == 1)).sum())
    fp = int((predicted & (labels == 0)).sum())
    fn = int((~predicted & (labels == 1)).sum())
    return {'loss': total / n_batches, 'pairs': len(labels), 'auc': auc,
            'brier': float(np.mean((probabilities - labels) ** 2)),
            'precision_at_0.5': tp / max(tp + fp, 1), 'recall_at_0.5': tp / max(tp + fn, 1)}


def train_verifier(cfg, train_cache, val_cache, output, epochs=10, batch_regions=16,
                   lr=1e-4, workers=0, margin=1.0, cf_weight=0.5, resume=None):
    if epochs < 1 or batch_regions < 1 or lr <= 0 or workers < 0 or margin < 0 or cf_weight < 0:
        raise ValueError('Invalid training options')
    train, val = RegionDataset(train_cache, 'train'), RegionDataset(val_cache, 'val')
    validate_training_caches(train, val)
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    seed_everything(cfg['seed'])
    settings = dict(cfg['model'], grid_size=cfg['grid_size'], feature_levels=2)
    spec = train.index['backend']
    for key in ('grid_size', 'context_scale', 'resize_short', 'resize_max'):
        if cfg[key] != spec[key]:
            raise ValueError(f'Config/cache mismatch: {key}')
    model = RegionTextVerifier(**settings).to(device)
    for data in (train, val):
        visual, _ = data[0]
        if visual.shape != (model.token_count, settings['dim']):
            raise ValueError('Model dimensions differ from cached visual feature dimensions')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    start, best, history = 0, float('inf'), []
    run_settings = {'batch_regions': batch_regions, 'lr': lr, 'margin': margin,
                    'cf_weight': cf_weight, 'seed': cfg['seed'], 'weight_decay': 1e-4}
    if resume:
        saved = load_tensor_file(resume)
        if (saved['train_cache_id'] != train.index['id'] or saved['val_cache_id'] != val.index['id']
                or saved['settings'] != settings or saved['run_settings'] != run_settings):
            raise ValueError('Resume configuration/cache differs from checkpoint')
        model.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        start, best, history = saved['epoch'] + 1, saved['best_val_loss'], saved['history']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if resume and Path(resume).resolve().parent != output.resolve():
        raise ValueError('Resume into the same checkpoint directory, preserving best.pt and history')
    if not resume and any((output / name).exists() for name in ('last.pt', 'best.pt')):
        raise ValueError('Training directory already contains checkpoints; choose a new output or use --resume')
    if start >= epochs:
        raise ValueError('epochs must exceed already completed epochs when resuming')
    for epoch in range(start, epochs):
        seed_everything(cfg['seed'] + epoch)
        generator = torch.Generator().manual_seed(cfg['seed'] + epoch)
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_regions, shuffle=True,
            collate_fn=collate_regions, num_workers=workers, generator=generator)
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_regions, shuffle=False,
            collate_fn=collate_regions, num_workers=workers)
        tr = epoch_pass(model, train_loader, device, optimizer, margin, cf_weight)
        va = epoch_pass(model, val_loader, device, None, margin, cf_weight)
        improved = va['loss'] < best
        best = min(best, va['loss'])
        history.append({'epoch': epoch, 'train': tr, 'val': va})
        payload = {'schema': 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'settings': settings, 'run_settings': run_settings, 'epoch': epoch,
                   'best_val_loss': best, 'history': history,
                   'train_cache_id': train.index['id'], 'val_cache_id': val.index['id'],
                   'train_classes': train.index['allowed_classes'], 'val_classes': val.index['allowed_classes'],
                   'dataset_fingerprint': train.index['dataset_fingerprint'], 'backend': spec}
        atomic_torch_save(output / 'last.pt', payload)
        if improved:
            atomic_torch_save(output / 'best.pt', payload)
        write_json(output / 'history.json', history)
        print(f"epoch={epoch + 1} train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} val_auc={va['auc']:.4f}", flush=True)
    return {'best': str(output / 'best.pt'), 'last': str(output / 'last.pt'), 'epochs_completed': epochs}
