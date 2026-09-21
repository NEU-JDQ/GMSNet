# GMSNet

PyTorch code for image-text crisis classification. The training pipeline combines
a ConvNeXt V2 visual encoder, a DeBERTa text encoder, bidirectional cross-modal
attention, gated fusion, and a classification head with multiple dropout branches.
Auxiliary classifiers provide supervision for the individual modalities.

## Installation

Run the following commands from this directory, preferably in a virtual environment:

```bash
python -m pip install torch torchvision
python -m pip install -r requirements.txt
python -m pip install sentencepiece
```

Use a PyTorch/torchvision build compatible with your CUDA environment when training
on GPUs. PyTorch and torchvision are not listed in `requirements.txt`;
SentencePiece supports the DeBERTa tokenizer. Dependency versions are not pinned.

## Data and pretrained models

The default paths are relative to the working directory:

```text
parent_directory/
|-- GMSNet/
|   |-- train.py
|   |-- test.py
|   |-- requirements.txt
|-- datasets/
|   |-- settingA/
|       |-- task_damage_text_img_train.tsv
|       |-- task_damage_text_img_dev.tsv
|       |-- task_damage_text_img_test.tsv
|       |-- data_image/...
|-- local_models/
    |-- convnextv2_base/
    |   |-- model.safetensors
    |-- deberta-v3-base/
        |-- ... model configuration, weights, and tokenizer files
```

Prepare the datasets and model files separately. Each TSV must contain
`tweet_text`, `image`, and `label` columns. Image paths are resolved relative to
the dataset root. Label strings must match `data/config.py`.

| Task argument | TSV name component | Classes in this code |
| --- | --- | --- |
| `task1` | `informative` | 2 |
| `task2` | `humanitarian` | 8 |
| `task3` | `damage` | 3 |

The filename pattern is `task_{name}_text_img_{split}.tsv`, where `split` is
`train`, `dev`, or `test`.

The visual encoder expects `../local_models/convnextv2_base/model.safetensors`.
If this file is absent, it starts with randomly initialized visual weights.
The current training entry does not pass its `--visual_weights` argument to the
encoder, so use the expected location. The text model location can be supplied
through `--text_model_path`.

## Training

The current training script selects `cuda:1` when CUDA is available, so this
configuration requires at least two visible GPUs. Device selection is hardcoded;
there is no command-line device option. The training loop also uses CUDA mixed
precision.

```bash
python train.py --task_name task3 --run_name baseline --data_root ../datasets/settingA --text_model_path ../local_models/deberta-v3-base --output_dir ./output_gmsnet --num_workers 0
```

The best checkpoint is selected by validation weighted F1 and saved as
`output_gmsnet/task3/baseline/best_model.pt`. Training logs are written to the
same run directory. Replace the task argument and prepare the corresponding TSV
files to train another task. Use `python train.py --help` for available arguments.

## Evaluation

```bash
python test.py --task_name task3 --checkpoint_path ./output_gmsnet/task3/baseline/best_model.pt --data_root ../datasets/settingA --text_model_path ../local_models/deberta-v3-base --output_dir ./test_results_GMSNet --num_workers 0
```

Use the same task and model dimensions as the training run. Evaluation averages
logits from the original and horizontally flipped images. Results are saved in
`test_results_GMSNet/task3/`, including `metrics_report.txt`, `prediction.csv`,
and `test_predictions_detailed.csv`. The report includes accuracy, macro F1,
weighted F1, and per-class metrics.

## Additional files

- `data/`: dataset loading, label mappings, and preprocessing.
- `modules/`: encoders, multimodal fusion, and classification heads.
- `extract_weights.py`: inspect modality gate values for an image-text pair.
- `attentionHeatmap.py`: visualize text-to-image attention.
- `inference_migcs.py`: inference wrapper returning probabilities and a heatmap.
- `check_data.py`: inspect label distributions and train/dev overlap.
- `new_train_dataset.py`: generate alternative splits by moving sampled test
  rows into training. This changes the held-out evaluation split and is not part
  of the training and evaluation commands above.

Utility scripts contain example or local paths; check their arguments or path
settings before use.
