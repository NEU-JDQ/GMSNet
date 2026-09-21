# GMSNet

ConvNeXt V2 + DeBERTa V3, bidirectional attention, independent sigmoid gates,
and unimodal auxiliary classification. Task 2 uses **eight classes**, in the
fixed order recorded in `data/config.py`.

## Installation

Use a compatible PyTorch/torchvision build for your CPU/CUDA environment:

```bash
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

Dependencies are not pinned to a fully verified training environment. Each run
records installed versions. Regression checks were run on CPU with PyTorch 2.8.0
and scikit-learn 1.6.1; this is not a full pretrained-model training reproduction.

## Preserve and audit the dataset

Prepare unmodified, verified train/dev/test TSVs and image files. The default
dataset and pretrained-model paths resolve relative to this project, not the
working directory:

```text
parent_directory/
|-- GMSNet/
|-- datasets/settingA/
|   |-- task_humanitarian_text_img_train.tsv
|   |-- task_humanitarian_text_img_dev.tsv
|   |-- task_humanitarian_text_img_test.tsv
|   |-- data_image/...
|-- local_models/
    |-- convnextv2_base/model.safetensors
    |-- deberta-v3-base/...
```

| Task | TSV name component | Declared classes |
| --- | --- | --- |
| task1 | informative | 2 |
| task2 | humanitarian | 8 |
| task3 | damage | 3 |

TSV columns must include `tweet_text`, `image`, and `label`. Keep `tweet_id` when
available to detect a tweet's multiple images crossing splits. Unknown labels,
malformed rows, empty splits, and missing images raise errors. Quoted tabs,
newlines, a UTF-8 BOM, and empty text fields are supported.

```bash
python check_data.py --task_name task2 --data_root ../datasets/settingA --hash-images
```

This command needs only Python. It checks all three split pairs and reports label
counts, absent classes, TSV SHA-256 hashes, duplicate image rows and shared text.
Cross-split image paths/tweet IDs cause failure. `--hash-images` also detects
byte-identical images under different names. Generic repeated text alone is a
warning sign, not proof of leakage. Perceptual duplicates are not detected.

Training and evaluation automatically run the same audit. They do not select
hyperparameters on the test split. `new_train_dataset.py` has been removed; its old test-to-training
transfer command can no longer run. Previously generated `_new.tsv` files are NOT repaired
or deleted: recover verified originals from the dataset release. Disjoint splits
and hashes cannot by themselves prove that files match the original benchmark.

**Eight-class results must not be compared directly with published five-class
results. Re-run baselines with the same label map and split files.**

## Training

```bash
python train.py --task_name task2 --run_name task2_8class_seed42 --data_root ../datasets/settingA --visual_weights ../local_models/convnextv2_base/model.safetensors --text_model_path ../local_models/deberta-v3-base --seed 42
```

Defaults: batch size 10, accumulation 8, maximum 40 epochs, encoder LR 5e-6,
new-layer LR 3e-5, weight decay .05, label smoothing .05, patience 4. Auxiliary
loss weight defaults to .1 for Task 2 and .05 for Tasks 1/3; `--lambda_aux 0`
remains a valid ablation. These are configuration defaults, not a claim that the
paper's scores have been reproduced with the corrected code.

Device defaults to the first available CUDA device, otherwise CPU. Override with
`--device cpu` or `--device cuda:1`. Windows worker transforms are picklable;
`--num_workers 0` is useful for diagnosing data problems.

An existing nonempty run directory is rejected. Choose a new `--run_name` for
every experiment. There is no automatic loading of old weights. Explicit
`--init_checkpoint path` is a weight-only warm start, not an exact resume; its
optimizer and scheduler start fresh. Missing/incompatible visual pretrained
weights fail instead of silently starting with a random encoder.

The best model is selected on validation `weighted_f1` by default. Use
`--selection_metric macro_f1` if that is the declared selection protocol, applying
the same choice to all baselines before looking at test performance. Both metrics
are logged. Outputs include `best_model.pt`, `train.log`, and `run_config.json`
with arguments, label mapping, split hashes/counts, versions and weight hashes.

## Evaluation

```bash
python test.py --task_name task2 --checkpoint_path ./output_gmsnet/task2/task2_8class_seed42/best_model.pt --data_root ../datasets/settingA --text_model_path ../local_models/deberta-v3-base
```

No test-time augmentation is used by default. Add `--tta` to average original and
horizontally flipped image logits; disclose this and use the same protocol for
baselines. The score and per-class report both use all declared classes with
`zero_division=0`. Thus absent classes contribute zero to eight-class Macro-F1.

Results in `test_results_GMSNet/task2/` include metrics, predictions with sample
image paths, and `evaluation_config.json`. Use different `--output_dir` values for
different checkpoints/TTA variants. Known checkpoint metadata is checked against
task, architecture and split hashes. Legacy raw state dictionaries remain readable
with a warning, but their training provenance cannot be verified.

## Compatibility and interpretation

Fusion now propagates the text padding mask through self-attention, cross-attention
and pooling. State-dict parameter names/shapes are unchanged, but predictions can
change. Re-train/re-evaluate before replacing scientific results. Gradient
accumulation now handles short final batches/windows by actual sample count; the
scheduler counts every optimizer update and skips AMP-overflow updates.

Image preprocessing still uses square padding, 224-pixel resize and the original
(.5, .5, .5) normalization. Check its compatibility with your pretrained weights
and disclose it; do not silently switch preprocessing for an existing checkpoint.
The original text cleaner removes complete hashtags; evaluate retaining hashtag
words if they carry useful crisis cues.

Gate percentages are normalized gate values, not confidence calibration or causal
modality contributions. Both pooled branches already contain cross-modal content.
The old visualization utilities remain illustrative: `inference_migcs.py` uses a
similarity proxy and `attentionHeatmap.py` probes a layer outside its full forward
path. Do not present these as exact attention/causal explanations without revising
and validating the visualization pipeline.

## Regression checks

```bash
python -m unittest discover -s tests -v
```

The suite covers held-out leakage guards, unchanged dataset files, strict TSV
handling, fixed eight-class metrics, mask invariance/gradients, model mask routing,
accumulation equivalence, scheduler counts, TTA defaults, missing visual weights,
metadata mismatches and Windows transform serialization. It needs torch,
torchvision, Pillow and scikit-learn for tensor/image checks, and no pretrained
models or downloaded datasets. Protocol checks can run without ML packages.
