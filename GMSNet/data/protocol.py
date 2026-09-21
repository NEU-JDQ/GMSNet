"""Read-only split validation and provenance (standard library only)."""
import csv
import hashlib
import os
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path, label_map):
    """Preserve quoted tabs/newlines; reject invalid records instead of dropping them."""
    rows = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t', strict=True)
        required = {'tweet_text', 'image', 'label'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: required columns are {sorted(required)}')
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f'{path}: duplicate column names')
        for row in reader:
            location = f'{path}:{reader.line_num}'
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'{location}: malformed TSV row; check quoting and column count')
            row['image'] = row['image'].strip()
            row['label'] = row['label'].strip()
            if not row['image']:
                raise ValueError(f'{location}: empty image path')
            if row['label'] not in label_map:
                raise ValueError(f"{location}: unknown label {row['label']!r}; use the declared label map")
            rows.append(row)
    if not rows:
        raise ValueError(f'{path}: split contains no samples')
    return rows


def resolve_image(root, value):
    return Path(root, value.replace('\\', '/')).resolve()


def audit_splits(root, task_info, hash_images=False):
    """Reject cross-split image paths/tweet IDs and optionally identical image bytes.

    TSV hashes record the supplied split, not proof that it is the official release.
    Repeated generic text is reported but is not alone treated as leakage.
    """
    root = Path(root).resolve()
    label_map = task_info['label_map']
    if sorted(label_map.values()) != list(range(task_info['num_classes'])):
        raise ValueError('Label IDs must be unique, contiguous and match num_classes')
    report = {'root': str(root), 'label_map': label_map,
              'image_content_hashing': hash_images, 'splits': {}, 'overlap': {}}
    identities = {}
    hash_cache = {}
    for split in ('train', 'dev', 'test'):
        path = root / f"task_{task_info['name']}_text_img_{split}.tsv"
        rows = read_rows(path, label_map)
        counts = Counter(row['label'] for row in rows)
        keys = {'image': set(), 'tweet_id': set(), 'text': set(), 'image_sha256': set()}
        for row in rows:
            image_path = resolve_image(root, row['image'])
            if not image_path.is_file():
                raise FileNotFoundError(f'{path}: image not found: {image_path}')
            image_key = os.path.normcase(str(image_path))
            keys['image'].add(image_key)
            for column in ('tweet_id', 'tweetid'):
                if row.get(column, '').strip():
                    keys['tweet_id'].add(row[column].strip())
            normalized_text = ' '.join(row['tweet_text'].split())
            if normalized_text:
                keys['text'].add(normalized_text)
            if hash_images:
                if image_key not in hash_cache:
                    hash_cache[image_key] = sha256_file(image_path)
                keys['image_sha256'].add(hash_cache[image_key])
        absent = [label for label in label_map if not counts[label]]
        if absent:
            warnings.warn(f'{path.name}: classes with zero support: {absent}. '
                          'Macro-F1 still uses the full declared label set.')
        report['splits'][split] = {
            'path': str(path), 'sha256': sha256_file(path), 'samples': len(rows),
            'class_counts': {label: counts[label] for label in label_map},
            'missing_classes': absent,
            'duplicate_image_rows': len(rows) - len(keys['image']),
            'tweet_ids_present': bool(keys['tweet_id']),
        }
        identities[split] = keys
    errors = []
    for left, right in combinations(identities, 2):
        pair = f'{left}/{right}'
        overlap = {key: len(identities[left][key] & identities[right][key])
                   for key in identities[left]}
        report['overlap'][pair] = overlap
        for key in ('image', 'tweet_id', 'image_sha256'):
            if overlap[key]:
                errors.append(f'{pair}: {overlap[key]} overlapping {key} identities')
    if errors:
        raise ValueError('Split overlap detected; do not train/evaluate. ' + '; '.join(errors)
                         + '. Restore verified source splits; do not silently delete test rows.')
    return report
