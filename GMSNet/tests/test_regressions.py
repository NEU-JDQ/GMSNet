"""Small CPU regression tests; no datasets, downloads or pretrained models needed.

Run: python -m unittest discover -s tests -v
Protocol tests only need Python; tensor/metric tests need torch and scikit-learn.
"""
import argparse
import ast
import copy
import csv
import importlib.util
import math
import os
import pickle
import sys
import tempfile
import unittest
import shutil
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.config import TASK_CONFIG
from data.protocol import audit_splits, read_rows, sha256_file


@contextmanager
def scratch_directory():
    # Inherit the Windows sandbox ACL; tempfile's explicit 0700 can remove it.
    base = Path(tempfile.gettempdir()).resolve()
    path = base / ('gmsnet-tests-' + uuid.uuid4().hex)
    path.mkdir(mode=0o777)
    try:
        yield str(path)
    finally:
        if path.resolve().parent != base or not path.name.startswith('gmsnet-tests-'):
            raise RuntimeError('Refusing cleanup outside the test temporary directory')
        shutil.rmtree(path)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_function(path, name, namespace):
    # Load the real function independently of optional pretrained-backbone imports.
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = scratch_directory()
        self.root = Path(self.temp.__enter__())
        self.addCleanup(self.temp.__exit__, None, None, None)
        self.task = TASK_CONFIG['task2']
        self.rows = {}
        for split in ('train', 'dev', 'test'):
            rows = []
            for i, label in enumerate(self.task['label_map']):
                image = f'{split}-{i}.jpg'
                (self.root / image).write_bytes(image.encode())
                rows.append({'tweet_text': f'{split} text {i}', 'image': image,
                             'label': label, 'tweet_id': f'{split}-{i}'})
            self.rows[split] = rows
            self.save(split)

    def path(self, split):
        return self.root / f'task_humanitarian_text_img_{split}.tsv'

    def save(self, split):
        with self.path(split).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['tweet_text', 'image', 'label', 'tweet_id'], delimiter='\t')
            writer.writeheader()
            writer.writerows(self.rows[split])

    def test_clean_eight_class_splits_are_read_only(self):
        before = {p.name: sha256_file(p) for p in self.root.iterdir()}
        report = audit_splits(self.root, self.task, hash_images=True)
        self.assertEqual(report['splits']['test']['samples'], 8)
        self.assertEqual(report['splits']['test']['missing_classes'], [])
        self.assertEqual(before, {p.name: sha256_file(p) for p in self.root.iterdir()})

    def test_train_test_image_overlap_is_blocked(self):
        self.rows['test'][0]['image'] = self.rows['train'][0]['image']
        self.save('test')
        with self.assertRaisesRegex(ValueError, 'train/test.*image'):
            audit_splits(self.root, self.task)

    def test_dev_test_tweet_overlap_is_blocked(self):
        self.rows['test'][0]['tweet_id'] = self.rows['dev'][0]['tweet_id']
        self.save('test')
        with self.assertRaisesRegex(ValueError, 'dev/test.*tweet_id'):
            audit_splits(self.root, self.task)

    def test_renamed_identical_images_are_detected_by_hash(self):
        (self.root / 'test-0.jpg').write_bytes((self.root / 'train-0.jpg').read_bytes())
        with self.assertRaisesRegex(ValueError, 'image_sha256'):
            audit_splits(self.root, self.task, hash_images=True)

    def test_unknown_label_does_not_disappear(self):
        self.rows['test'][0]['label'] = 'misspelled_label'
        self.save('test')
        with self.assertRaisesRegex(ValueError, 'unknown label'):
            read_rows(self.path('test'), self.task['label_map'])

    def test_quoted_tabs_newlines_and_empty_text_are_preserved(self):
        self.rows['test'][0]['tweet_text'] = 'a\tb\nquoted "text"'
        self.rows['test'][1]['tweet_text'] = ''
        self.save('test')
        rows = read_rows(self.path('test'), self.task['label_map'])
        self.assertEqual(rows[0]['tweet_text'], self.rows['test'][0]['tweet_text'])
        self.assertEqual(rows[1]['tweet_text'], '')
        self.assertEqual(len(rows), 8)

    def test_malformed_row_is_rejected(self):
        with self.path('test').open('a', encoding='utf-8') as stream:
            stream.write('too\tfew\n')
        with self.assertRaisesRegex(ValueError, 'malformed'):
            read_rows(self.path('test'), self.task['label_map'])

    def test_missing_image_is_rejected(self):
        (self.root / 'test-0.jpg').unlink()
        with self.assertRaises(FileNotFoundError):
            audit_splits(self.root, self.task)

    def test_retired_transfer_cannot_write(self):
        legacy_path = ROOT / 'new_train_dataset.py'
        if not legacy_path.exists():
            # Removing the obsolete entry point also prevents legacy execution.
            return
        retired = load_file('retired_transfer', legacy_path)
        before = {p.name: sha256_file(p) for p in self.root.iterdir()}
        with self.assertRaisesRegex(RuntimeError, 'disabled'):
            retired.process_datasets(str(self.root), 0.45)
        self.assertEqual(before, {p.name: sha256_file(p) for p in self.root.iterdir()})


try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from experiment_utils import classification_metrics, check_checkpoint_manifest, write_manifest
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class QuietProgress:
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, **kwargs):
        pass


@unittest.skipUnless(ML_AVAILABLE, 'requires torch and scikit-learn')
class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(123)

    def test_macro_f1_includes_all_eight_classes(self):
        metrics, report = classification_metrics([0, 1, 1], [0, 1, 1], TASK_CONFIG['task2']['label_map'])
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertAlmostEqual(metrics['macro_f1'], 0.25)
        self.assertIn('0.2500', report)

    def test_transforms_are_picklable_for_windows_workers(self):
        from data.transforms import get_transforms
        from PIL import Image
        transform = pickle.loads(pickle.dumps(get_transforms('eval')))
        output = transform(Image.new('RGB', (31, 17)))
        self.assertEqual(tuple(output.shape), (3, 224, 224))

    def test_missing_visual_weights_fail_before_model_allocation(self):
        path = ROOT / 'modules/encoders.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == 'ConvNextVisualEncoder')
        namespace = {'nn': nn, 'os': os, 'Path': Path, '__file__': str(path)}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
        with self.assertRaisesRegex(FileNotFoundError, 'Visual pretrained weights'):
            namespace['ConvNextVisualEncoder'](weights_path=str(ROOT / 'does-not-exist.safetensors'))

    def test_model_passes_text_mask_to_fusion(self):
        model_cls = load_file('model_test', ROOT / 'modules/model.py').ModularCrisisModel

        class TextEncoder(nn.Module):
            def forward(self, inputs):
                return inputs['features']

        class Fusion(nn.Module):
            def forward(self, visual, text, text_attention_mask=None):
                self.mask = text_attention_mask
                return visual

        fusion = Fusion()
        model = model_cls(nn.Identity(), TextEncoder(), fusion, nn.Identity())
        mask = torch.tensor([[1, 1, 0]])
        model({'image': torch.randn(1, 4), 'text_tokens': {'features': torch.randn(1, 3, 4),
                                                         'attention_mask': mask}})
        self.assertIs(fusion.mask, mask)

    def test_padding_cannot_change_fused_output_or_receive_gradient(self):
        fusion_class = load_file('fusion_test', ROOT / 'modules/fusion.py').AGMFusion
        fusion = fusion_class(4, 4, embed_dim=8, num_heads=2).eval()
        visual = torch.randn(2, 3, 4)
        text = torch.randn(2, 6, 4, requires_grad=True)
        mask = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]])
        padded_changed = text.detach().clone()
        padded_changed[mask == 0] = 10000 * torch.randn_like(padded_changed[mask == 0])
        result = fusion(visual, text, mask)[0]
        with torch.no_grad():
            changed = fusion(visual, padded_changed, mask)[0]
        torch.testing.assert_close(result, changed, atol=2e-5, rtol=2e-5)
        result.square().sum().backward()
        torch.testing.assert_close(text.grad[mask == 0], torch.zeros_like(text.grad[mask == 0]))

    def test_all_padding_is_rejected(self):
        cls = load_file('fusion_empty', ROOT / 'modules/fusion.py').AGMFusion
        with self.assertRaisesRegex(ValueError, 'valid text token'):
            cls(4, 4, embed_dim=8, num_heads=2)(torch.randn(1, 3, 4), torch.randn(1, 6, 4), torch.zeros(1, 6))

    def test_accumulation_matches_large_batches_including_remainder(self):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 8)

            def forward(self, inputs, return_features=False):
                out = self.fc(inputs['image'])
                return (out, out, out) if return_features else out

        class Scheduler:
            def __init__(self):
                self.steps = 0

            def step(self):
                self.steps += 1

        train_epoch = load_function(ROOT / 'train.py', 'train_epoch', {'torch': torch, 'tqdm': QuietProgress})
        rows = [{'image': torch.randn(4), 'text_tokens': {'attention_mask': torch.ones(1)},
                 'label': torch.tensor(i % 8)} for i in range(7)]
        left = ToyModel()
        right = copy.deepcopy(left)
        for model, batch_size, accumulation in ((left, 2, 3), (right, 6, 1)):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scheduler = Scheduler()
            scaler = torch.cuda.amp.GradScaler(enabled=False)
            train_epoch(model, DataLoader(rows, batch_size=batch_size), optimizer,
                        nn.CrossEntropyLoss(), torch.device('cpu'), 1, 0.1, accumulation, scheduler, scaler)
            self.assertEqual(scheduler.steps, 2)
        for actual, expected in zip(left.parameters(), right.parameters()):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_tta_is_explicit(self):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, inputs):
                self.calls += 1
                score = inputs['image'].sum(dim=(1, 2, 3))
                return torch.stack([score, -score], dim=1)

        run_test = load_function(ROOT / 'test.py', 'run_test', {'torch': torch, 'tqdm': QuietProgress})
        rows = [{'image': torch.ones(3, 2, 2), 'text_tokens': {'attention_mask': torch.ones(1)},
                 'label': torch.tensor(0)} for _ in range(3)]
        model = ToyModel()
        run_test(model, DataLoader(rows, batch_size=2), torch.device('cpu'))
        self.assertEqual(model.calls, 2)
        model.calls = 0
        run_test(model, DataLoader(rows, batch_size=2), torch.device('cpu'), tta=True)
        self.assertEqual(model.calls, 4)

    def test_task_specific_training_defaults(self):
        parse_args = load_function(ROOT / 'train.py', 'parse_args', {'argparse': argparse,
                                  'DEFAULT_DATA_ROOT': 'dummy', 'PROJECT_ROOT': ROOT})
        with patch.object(sys, 'argv', ['train.py', '--task_name', 'task2']):
            args = parse_args()
        self.assertEqual((args.batch_size, args.accumulation_steps, args.epochs, args.lr, args.lambda_aux),
                         (10, 8, 40, 3e-5, 0.1))
        self.assertEqual(args.device, 'auto')
        with patch.object(sys, 'argv', ['train.py', '--task_name', 'task2', '--lambda_aux', '0']):
            self.assertEqual(parse_args().lambda_aux, 0)

    def test_schedule_counts_final_partial_window(self):
        tree = ast.parse((ROOT / 'train.py').read_text(encoding='utf-8'))
        assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'total_steps' for t in n.targets))
        count = eval(compile(ast.Expression(assignment.value), '<schedule>', 'eval'),
                     {'math': math, 'train_loader': range(4), 'args': SimpleNamespace(accumulation_steps=3, epochs=2)})
        self.assertEqual(count, 4)

    def test_changed_split_is_rejected_for_recorded_checkpoint(self):
        with scratch_directory() as temp:
            checkpoint = Path(temp) / 'best_model.pt'
            args = SimpleNamespace(embed_dim=8, num_heads=2, layers=1)
            audit = {'splits': {split: {'sha256': split} for split in ('train', 'dev', 'test')}}
            write_manifest(Path(temp) / 'run_config.json', args, TASK_CONFIG['task2'], audit)
            check_checkpoint_manifest(checkpoint, args, TASK_CONFIG['task2'], audit)
            changed = copy.deepcopy(audit)
            changed['splits']['test']['sha256'] = 'different'
            with self.assertRaisesRegex(ValueError, 'test TSV changed'):
                check_checkpoint_manifest(checkpoint, args, TASK_CONFIG['task2'], changed)


if __name__ == '__main__':
    unittest.main()
