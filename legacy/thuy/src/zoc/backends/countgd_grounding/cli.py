from __future__ import annotations

import argparse
import json
from pathlib import Path
from .io import load_config
from .fsc import FSC147


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be >= 1')
    return value


def main(argv=None):
    p = argparse.ArgumentParser(description='FSC-147 region-text grounding research pipeline')
    p.add_argument('--config', required=True, help='JSON paths are relative to the config file')
    sub = p.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('audit', help='Validate splits, metadata and actual image coordinates')
    audit.add_argument('--limit', type=positive_int, help='Check only N images per split for smoke testing')
    manifest = sub.add_parser('manifest', help='Create query manifest; other classes remain unknown')
    manifest.add_argument('--split', choices=['train','val','test'], required=True)
    manifest.add_argument('--output', required=True)
    manifest.add_argument('--negatives', type=int, default=4, help='Number of other-class probes; not automatic negative labels')
    manifest.add_argument('--all-queries', action='store_true', help='All class prompts of this split')
    manifest.add_argument('--limit', type=positive_int)
    manifest.add_argument('--overrides', help='Reviewed supplemental full-image point/absence annotations')
    regions = sub.add_parser('regions', help='Prepare exemplar ROI labels and pending negative prompts')
    regions.add_argument('--split', choices=['train','val'], required=True)
    regions.add_argument('--output', required=True)
    regions.add_argument('--negatives', type=int, default=4)
    regions.add_argument('--limit', type=positive_int)
    review = sub.add_parser('review', help='Generate standalone review HTML')
    review.add_argument('--kind', choices=['regions','queries'], required=True)
    review.add_argument('--input', required=True)
    review.add_argument('--output', required=True)
    review.add_argument('--offset', type=int, default=0)
    review.add_argument('--limit', type=positive_int, default=100)
    cache = sub.add_parser('cache', help='Extract fixed ROI/text features from verified region labels')
    cache.add_argument('--split', choices=['train','val'], required=True)
    cache.add_argument('--reviewed', required=True)
    cache.add_argument('--output', required=True)
    train = sub.add_parser('train', help='Train verifier on train cache; choose checkpoint on val cache')
    train.add_argument('--train-cache', required=True)
    train.add_argument('--val-cache', required=True)
    train.add_argument('--output', required=True)
    train.add_argument('--epochs', type=positive_int, default=10)
    train.add_argument('--batch-regions', type=positive_int, default=16)
    train.add_argument('--lr', type=float, default=1e-4)
    train.add_argument('--workers', type=int, default=0)
    train.add_argument('--margin', type=float, default=1.0)
    train.add_argument('--cf-weight', type=float, default=0.5)
    train.add_argument('--resume')
    infer = sub.add_parser('infer', help='Single-pass text-only baseline and optional verifier')
    infer.add_argument('--manifest', required=True)
    infer.add_argument('--output', required=True)
    infer.add_argument('--verifier')
    infer.add_argument('--limit', type=positive_int)
    infer.add_argument('--check-hooks', action='store_true', help='Check first newly computed prediction against unhooked forward')
    evaluate = sub.add_parser('evaluate', help='Evaluate only verified GT; record unknown queries separately')
    evaluate.add_argument('--manifest', required=True)
    evaluate.add_argument('--predictions', required=True)
    evaluate.add_argument('--output', required=True)
    evaluate.add_argument('--baseline-threshold', type=float)
    evaluate.add_argument('--pool-threshold', type=float)
    evaluate.add_argument('--grounding-threshold', type=float)
    evaluate.add_argument('--allow-partial', action='store_true')
    doctor = sub.add_parser('doctor', help='Real checkpoint forward and feature-hook smoke check')
    doctor.add_argument('--split', choices=['train','val'], default='val')
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == 'train':
        from .train import train_verifier
        result = train_verifier(cfg, args.train_cache, args.val_cache, args.output, args.epochs,
            args.batch_regions, args.lr, args.workers, args.margin, args.cf_weight, args.resume)
    else:
        ds = FSC147(cfg)
        if args.command == 'audit':
            result = ds.audit(args.limit)
        elif args.command == 'manifest':
            from .protocol import make_manifest
            result = make_manifest(ds, args.split, args.output, args.negatives, args.all_queries, args.limit, args.overrides)
        elif args.command == 'regions':
            from .protocol import make_region_review
            result = make_region_review(ds, args.split, args.output, args.negatives, args.limit)
        elif args.command == 'review':
            from .review import render_review
            result = render_review(ds, args.input, args.output, args.kind, args.offset, args.limit)
        elif args.command == 'evaluate':
            from .metrics import evaluate
            result = evaluate(ds, args.manifest, args.predictions, args.output, args.baseline_threshold,
                              args.pool_threshold, args.grounding_threshold, args.allow_partial)
        else:
            from .backend import CountGDBackend
            backend = CountGDBackend(cfg)
            if args.command == 'cache':
                from .cache import build_cache
                result = build_cache(ds, backend, args.reviewed, args.output, args.split)
            elif args.command == 'infer':
                from .infer import predict_manifest
                result = predict_manifest(ds, backend, args.manifest, args.output, args.verifier, args.limit, args.check_hooks)
            elif args.command == 'doctor':
                import torch
                from .model import RegionTextVerifier
                from .infer import score_proposals
                name = ds.splits[args.split][0]
                ds.validate_image(name)
                pred = backend.predict(ds.image_path(name), ds.classes[name], check_hooks=True)
                model = RegionTextVerifier(**dict(cfg['model'], grid_size=cfg['grid_size'],feature_levels=2)).to(backend.device).eval()
                ground, usable = score_proposals(backend, model, pred, ds.classes[name], cfg['roi_chunk'])
                result = {'image': name, 'query': ds.classes[name], 'native_queries': len(pred['native_scores']),
                          'usable_rois': int(usable.sum()), 'verifier_shape': list(ground.shape),
                          'finite': bool(torch.isfinite(ground).all()), 'hook_forward_equal': True,
                          'backend': backend.spec, 'note': 'Verifier here is RANDOM, only a shape/forward smoke test'}
            else:
                raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
