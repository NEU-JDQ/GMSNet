"""Run provenance and shared classification metrics."""
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score


def classification_metrics(true_labels, pred_labels, label_map):
    labels = list(range(len(label_map)))
    if len(true_labels) == 0 or len(true_labels) != len(pred_labels):
        raise ValueError('Expected non-empty, equally sized labels and predictions')
    if not set(true_labels).union(pred_labels).issubset(labels):
        raise ValueError('Found label IDs outside the declared task label set')
    names = [name for name, idx in sorted(label_map.items(), key=lambda item: item[1])]
    metrics = {'accuracy': accuracy_score(true_labels, pred_labels),
               'macro_f1': f1_score(true_labels, pred_labels, labels=labels,
                                    average='macro', zero_division=0),
               'weighted_f1': f1_score(true_labels, pred_labels, labels=labels,
                                       average='weighted', zero_division=0)}
    report = classification_report(true_labels, pred_labels, labels=labels,
                                   target_names=names, digits=4, zero_division=0)
    return metrics, report


def write_manifest(path, args, task_info, split_audit, **extra):
    packages = {}
    for package in ('torch', 'torchvision', 'timm', 'transformers', 'scikit-learn', 'numpy'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    payload = {'args': vars(args), 'task': task_info, 'split_audit': split_audit,
               'python': platform.python_version(), 'packages': packages,
               'fusion_text_padding_mask': True,
               'macro_f1_label_policy': 'all_declared_classes_zero_division_0', **extra}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def check_checkpoint_manifest(checkpoint, args, task_info, split_audit):
    """Verify known metadata; legacy checkpoints remain readable with an explicit warning."""
    import warnings
    path = Path(checkpoint).parent / 'run_config.json'
    if not path.exists():
        warnings.warn('Legacy checkpoint has no run_config.json. Its training split and label '
                      'mapping cannot be verified; padding-mask fixes may change predictions.')
        return
    config = json.loads(path.read_text(encoding='utf-8'))
    if config['task'] != task_info:
        raise ValueError('Checkpoint task/label mapping differs from the evaluation task')
    for name in ('embed_dim', 'num_heads', 'layers'):
        if config['args'][name] != getattr(args, name):
            raise ValueError(f'Checkpoint architecture mismatch: {name}')
    for split in ('train', 'dev', 'test'):
        if config['split_audit']['splits'][split]['sha256'] != split_audit['splits'][split]['sha256']:
            raise ValueError(f'{split} TSV changed since training; evaluate the recorded splits')
