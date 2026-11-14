from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import torch
from .io import read_jsonl, write_json, read_json, atomic_torch_save, load_tensor_file, digest, file_hash
from .protocol import checked_region_labels


def build_cache(ds, backend, review_path, cache_dir, split):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    seen, queries = set(), set()
    for row in read_jsonl(review_path):
        if row['id'] in seen:
            raise ValueError(f"Duplicate region ID: {row['id']}")
        seen.add(row['id'])
        labels = checked_region_labels(ds, row, split)
        if not labels:
            continue
        grouped[row['image']].append((row, labels))
        queries.update(x['query'] for x in labels)
    if not grouped:
        raise ValueError('No verified region labels; review labels first')
    text_bank = {q: backend.encode_text(q).cpu() for q in sorted(queries)}
    text_path = cache_dir / 'texts.pt'
    atomic_torch_save(text_path, {'backend_id': backend.spec['id'], 'texts': text_bank})
    entries = []
    for number, (image, rows) in enumerate(grouped.items(), 1):
        image_hash = file_hash(ds.image_path(image))
        needs = []
        for row, labels in rows:
            key = digest([backend.spec['id'], ds.fingerprint, image_hash, row['box_xyxy']])
            relative = 'features/' + key + '.pt'
            path = cache_dir / relative
            if not path.is_file():
                needs.append((row, path))
            entries.append({'id': row['id'], 'image': image, 'box_xyxy': row['box_xyxy'],
                            'image_sha256': image_hash, 'feature': relative, 'labels': labels})
        if needs:
            features, resized_size, original_size = backend.features(ds.image_path(image))
            for start in range(0, len(needs), ds.cfg['roi_chunk']):
                chunk = needs[start:start + ds.cfg['roi_chunk']]
                tokens = backend.tokens(features, [r['box_xyxy'] for r, _ in chunk], resized_size, original_size)
                for value, (row, path) in zip(tokens, chunk):
                    value = value.cpu().half()
                    if not torch.isfinite(value).all():
                        raise ValueError('ROI features overflowed float16 cache; aborting')
                    atomic_torch_save(path, {'backend_id': backend.spec['id'], 'visual': value})
        for entry in entries[-len(rows):]:
            entry['feature_sha256'] = file_hash(cache_dir / entry['feature'])
        if number % 10 == 0 or number == len(grouped):
            print(f'cache {split}: {number}/{len(grouped)} images', flush=True)
    result = {'schema': 1, 'split': split, 'dataset_fingerprint': ds.fingerprint,
              'allowed_classes': sorted(ds.class_sets[split]), 'backend': backend.spec,
              'entries': entries, 'text_file': 'texts.pt', 'text_sha256': file_hash(text_path)}
    result['id'] = digest(result)
    write_json(cache_dir / 'index.json', result)
    counts = {q: {'positive': 0, 'negative': 0} for q in queries}
    for row in entries:
        for label in row['labels']:
            counts[label['query']]['positive' if label['label'] == 1 else 'negative'] += 1
    return {'regions': len(entries), 'queries': len(queries),
            'positives': sum(x['label'] == 1 for r in entries for x in r['labels']),
            'negatives': sum(x['label'] == 0 for r in entries for x in r['labels']),
            'index': str(cache_dir / 'index.json'), 'per_query_label_counts': counts,
            'one_sided_queries': sorted(q for q, n in counts.items() if not all(n.values()))}


class RegionDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir, expected_split=None):
        self.root = Path(cache_dir)
        self.index = read_json(self.root / 'index.json')
        check = dict(self.index)
        recorded = check.pop('id')
        if digest(check) != recorded:
            raise ValueError('Cache index modified; rerun cache after editing review labels')
        if expected_split and self.index['split'] != expected_split:
            raise ValueError(f"Expected {expected_split} cache; got {self.index['split']}")
        path = self.local_path(self.index['text_file'])
        if file_hash(path) != self.index['text_sha256']:
            raise ValueError('Text cache checksum mismatch')
        bank = load_tensor_file(path)
        if bank['backend_id'] != self.index['backend']['id']:
            raise ValueError('Text cache backend mismatch')
        self.texts = bank['texts']
        self.entries = self.index['entries']
        self.verified_files = set()
        if not self.entries:
            raise ValueError('Empty cache')
        for row in self.entries:
            if not row['labels']:
                raise ValueError('Cache contains empty region supervision')
            for item in row['labels']:
                if item['query'] not in self.index['allowed_classes'] or item['query'] not in self.texts:
                    raise ValueError('Cache query outside permitted classes/text bank')
                if type(item['label']) is not int or item['label'] not in (0, 1):
                    raise ValueError('Cache contains unverified labels')

    def local_path(self, relative):
        path = (self.root / relative).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError('Cache path escapes cache directory')
        return path

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        row = self.entries[index]
        path = self.local_path(row['feature'])
        if row['feature'] not in self.verified_files:
            if file_hash(path) != row['feature_sha256']:
                raise ValueError('ROI cache checksum mismatch')
            self.verified_files.add(row['feature'])
        saved = load_tensor_file(path)
        if saved['backend_id'] != self.index['backend']['id']:
            raise ValueError('ROI cache backend mismatch')
        visual = saved['visual'].float()
        if visual.ndim != 2 or not torch.isfinite(visual).all():
            raise ValueError('Invalid visual cache')
        return visual, [(self.texts[x['query']].float(), x['label']) for x in row['labels']]


def collate_regions(samples):
    visual, text, labels, region_ids = [], [], [], []
    for group, (roi, pairs) in enumerate(samples):
        for tokens, label in pairs:
            if tokens.ndim != 2 or len(tokens) == 0 or not torch.isfinite(tokens).all():
                raise ValueError('Invalid text tokens')
            visual.append(roi)
            text.append(tokens)
            labels.append(label)
            region_ids.append(group)
    if not text:
        raise ValueError('Empty collate batch')
    padded = torch.nn.utils.rnn.pad_sequence(text, batch_first=True)
    lengths = torch.tensor([len(t) for t in text])
    valid = torch.arange(padded.shape[1])[None, :] < lengths[:, None]
    return {'visual': torch.stack(visual), 'text': padded, 'text_valid': valid,
            'labels': torch.tensor(labels, dtype=torch.float32),
            'region_ids': torch.tensor(region_ids, dtype=torch.long)}
