
import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, f1_score


from data import CrisisDataset, TextProcessor, get_transforms, DEFAULT_DATA_ROOT, TASK_CONFIG


from modules import (
    AGMFusion,
    CrisisKANClassifier,
    ModularCrisisModel,
    ConvNextVisualEncoder,
    DebertaTextEncoder
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GMSNet")

    parser.add_argument('--task_name', type=str, default='task3', choices=['task1', 'task2', 'task3'])
    parser.add_argument('--checkpoint_path', type=str,
                        default='./output_gmsnet/task3/task3/best_model.pt',
                        help='Path to the trained best_model.pt checkpoint')
    parser.add_argument('--output_dir', type=str, default='./test_results_GMSNet')

    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)


    parser.add_argument('--text_model_path', type=str, default='../local_models/deberta-v3-base')

    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.4)

    return parser.parse_args()


def run_test(model, loader, device):
    model.eval()
    model = model.float()
    all_preds = []
    all_labels = []

    print(f"Evaluating {len(loader.dataset)} samples...")

    with torch.no_grad():
        loop = tqdm(loader, desc="Testing GMSNet")
        for batch in loop:
            images = batch['image'].to(device)
            text_inputs = {k: v.to(device) for k, v in batch['text_tokens'].items()}
            labels = batch['label'].to(device)

            inputs = {'image': images, 'text_tokens': text_inputs}

            # Average original and horizontally flipped image logits for test-time augmentation.
            logits_normal = model(inputs)

            flipped_images = torch.flip(images, dims=[3])  
            inputs_flipped = {'image': flipped_images, 'text_tokens': text_inputs}
            logits_flipped = model(inputs_flipped)

            logits = (logits_normal + logits_flipped) / 2.0

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading the test dataset...")
    text_proc = TextProcessor(model_name=args.text_model_path)
    eval_transform = get_transforms(mode='eval')

    test_set = CrisisDataset(args.data_root, args.task_name, 'test', eval_transform, text_proc)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = TASK_CONFIG[args.task_name]['num_classes']
    label_map = TASK_CONFIG[args.task_name]['label_map']
    id_to_label = {v: k for k, v in label_map.items()}
    target_names = [id_to_label[i] for i in range(num_classes)]

    print("Initializing GMSNet components...")
    vis_enc = ConvNextVisualEncoder()
    txt_enc = DebertaTextEncoder(model_path=args.text_model_path)
    fusion = AGMFusion(
        visual_dim=vis_enc.output_dim,
        text_dim=txt_enc.output_dim,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        layers=args.layers
    )

    cls_head = CrisisKANClassifier(
        input_dim=fusion.output_dim,
        num_classes=num_classes,
        dropout_rate=args.dropout
    )

    model = ModularCrisisModel(vis_enc, txt_enc, fusion, cls_head,
                               num_classes=num_classes, embed_dim=args.embed_dim)
    model.to(device)

    print(f"Loading checkpoint from: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path} \nRun train.py to generate a checkpoint, or provide an existing checkpoint path.")

    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Model weights loaded.")

    true_labels, pred_labels = run_test(model, test_loader, device)

    acc = accuracy_score(true_labels, pred_labels)
    weighted_f1 = f1_score(true_labels, pred_labels, average='weighted')
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')

    report_str = f"Accuracy:    {acc:.4f}\n"
    report_str += f"Macro F1:    {macro_f1:.4f}\n"
    report_str += f"Weighted F1: {weighted_f1:.4f}\n"
    report_str += "-" * 50 + "\n"
    report_str += classification_report(true_labels, pred_labels, labels=list(range(num_classes)),
                                        target_names=target_names, digits=4)
    report_str += "\n" + "=" * 50

    print("\n" + "=" * 50)
    print(f"GMSNet evaluation report ({args.task_name.upper()})")
    print("=" * 50)
    print(report_str)

    final_output_dir = os.path.join(args.output_dir, args.task_name)
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)

    metrics_path = os.path.join(final_output_dir, 'metrics_report.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"GMSNet evaluation report ({args.task_name.upper()})\n")
        f.write("=" * 50 + "\n")
        f.write(report_str + "\n")
    print(f"Metrics report saved to: {metrics_path}")

    pred_path = os.path.join(final_output_dir, 'prediction.csv')
    with open(pred_path, 'w') as f:
        for p in pred_labels:
            f.write(f"{p}\n")
    print(f"Predictions saved to: {pred_path}")

    df = pd.DataFrame({
        'True Label ID': true_labels,
        'Pred Label ID': pred_labels,
        'True Label Name': [id_to_label[i] for i in true_labels],
        'Pred Label Name': [id_to_label[i] for i in pred_labels]
    })
    detailed_path = os.path.join(final_output_dir, 'test_predictions_detailed.csv')
    df.to_csv(detailed_path, index=False)
    print(f"Detailed results saved to: {detailed_path}")

if __name__ == "__main__":
    main()