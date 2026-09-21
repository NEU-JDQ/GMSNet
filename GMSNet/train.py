import os
import math
from pathlib import Path
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from data.protocol import audit_splits, sha256_file
from data.config import PROJECT_ROOT
from experiment_utils import classification_metrics, write_manifest

from data import CrisisDataset, TextProcessor, get_transforms, DEFAULT_DATA_ROOT, TASK_CONFIG


from modules import (
    ConvNextVisualEncoder,
    DebertaTextEncoder,
    AGMFusion,  
    CrisisKANClassifier,
    ModularCrisisModel
)






def parse_args():
    parser = argparse.ArgumentParser(description="Train GMSNet")
    parser.add_argument('--task_name', type=str, default='task3', choices=['task1', 'task2', 'task3'])
    parser.add_argument('--run_name', type=str, default='task3', help='Experiment identifier')
    parser.add_argument('--output_dir', type=str, default='./output_gmsnet', help='Root directory for experiment outputs')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--accumulation_steps', type=int, default=8,
                        help='Gradient accumulation steps (effective batch size = batch_size * steps)')

    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--patience', type=int, default=4)

    parser.add_argument('--lambda_aux', type=float, default=None, help='Weight applied to the sum of the auxiliary classification losses')

    parser.add_argument('--visual_weights', type=str, default=str(PROJECT_ROOT.parent / 'local_models/convnextv2_base/model.safetensors'))
    parser.add_argument('--text_model_path', type=str, default=str(PROJECT_ROOT.parent / 'local_models/deberta-v3-base'))
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.4)

    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--selection_metric', choices=['macro_f1', 'weighted_f1'], default='weighted_f1')
    parser.add_argument('--hash-images', action='store_true')
    parser.add_argument('--init_checkpoint', default=None,
                        help='Explicit warm start of weights only; optimizer/scheduler start fresh')
    args = parser.parse_args()
    if args.lambda_aux is None:
        args.lambda_aux = 0.1 if args.task_name == 'task2' else 0.05
    if min(args.batch_size, args.accumulation_steps, args.epochs, args.patience) < 1:
        parser.error('batch_size, accumulation_steps, epochs and patience must be positive')
    if args.lr <= 0 or args.lambda_aux < 0:
        parser.error('lr must be positive and lambda_aux must be non-negative')
    return args



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def setup_logger(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[logging.FileHandler(os.path.join(output_dir, 'train.log')), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)






def train_epoch(model, loader, optimizer, criterion_ce, device, epoch,
                lambda_aux, accumulation_steps, scheduler, scaler):
    model.train()
    totals = {'loss': 0.0, 'correct': 0, 'samples': 0}
    accumulated_samples = 0
    optimizer.zero_grad(set_to_none=True)
    loop = tqdm(loader, desc=f'Train Epoch {epoch}')
    for i, batch in enumerate(loop):
        images = batch['image'].to(device)
        text_inputs = {k: v.to(device) for k, v in batch['text_tokens'].items()}
        labels = batch['label'].to(device)
        count = labels.size(0)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            logits, aux_v, aux_t = model({'image': images, 'text_tokens': text_inputs}, return_features=True)
            loss_ce = criterion_ce(logits, labels)
            loss_aux = criterion_ce(aux_v, labels) + criterion_ce(aux_t, labels)
            loss = loss_ce + lambda_aux * loss_aux
        # Sum per-example gradients; divide by the actual window sample count.
        # This also handles both incomplete accumulation windows and short batches.
        scaler.scale(loss * count).backward()
        accumulated_samples += count
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_samples)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()  # AMP overflow must not advance the LR schedule.
            optimizer.zero_grad(set_to_none=True)
            accumulated_samples = 0
        totals['loss'] += loss.item() * count
        totals['correct'] += (logits.argmax(dim=1) == labels).sum().item()
        totals['samples'] += count
        loop.set_postfix(loss=loss.item(), acc=totals['correct'] / totals['samples'])
    return totals['loss'] / totals['samples'], totals['correct'] / totals['samples']


def evaluate(model, loader, criterion_ce, device, lambda_aux, label_map):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            text_inputs = {k: v.to(device) for k, v in batch['text_tokens'].items()}
            labels = batch['label'].to(device)
            logits, aux_v, aux_t = model({'image': images, 'text_tokens': text_inputs}, return_features=True)
            loss = criterion_ce(logits, labels) + lambda_aux * (
                criterion_ce(aux_v, labels) + criterion_ce(aux_t, labels))
            total_loss += loss.item() * labels.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    metrics, _ = classification_metrics(all_labels, all_preds, label_map)
    return total_loss / len(all_labels), metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    save_dir = os.path.join(args.output_dir, args.task_name, args.run_name)
    if Path(save_dir).exists() and any(Path(save_dir).iterdir()):
        raise FileExistsError(f'Run directory is not empty: {save_dir}. Choose a new --run_name. '
                              'Old weights are never loaded implicitly.')
    task_info = TASK_CONFIG[args.task_name]
    split_audit = audit_splits(args.data_root, task_info, args.hash_images)
    logger = setup_logger(save_dir)
    logger.info(f"Starting GMSNet training: {args.run_name}")

    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices: cpu and cuda')
    if device.type == 'cuda' and (not torch.cuda.is_available() or
            (device.index is not None and device.index >= torch.cuda.device_count())):
        raise ValueError(f'CUDA device is not available: {device}')
    logger.info(f"Device: {device}")

    logger.info("Loading training and validation datasets...")
    text_proc = TextProcessor(model_name=args.text_model_path)
    train_transform = get_transforms(mode='train')
    eval_transform = get_transforms(mode='eval')

    train_set = CrisisDataset(args.data_root, args.task_name, 'train', train_transform, text_proc)
    dev_set = CrisisDataset(args.data_root, args.task_name, 'dev', eval_transform, text_proc)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = TASK_CONFIG[args.task_name]['num_classes']

    logger.info("Initializing GMSNet components...")
    vis_enc = ConvNextVisualEncoder(weights_path=None if args.init_checkpoint else args.visual_weights)
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
    if args.init_checkpoint:
        logger.warning('Explicit weight-only warm start; not an exact training resume: %s', args.init_checkpoint)
        model.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'), strict=True)
    model.to(device)
    write_manifest(Path(save_dir) / 'run_config.json', args, task_info, split_audit,
                   visual_weights_sha256=sha256_file(args.visual_weights) if not args.init_checkpoint else None,
                   init_checkpoint_sha256=sha256_file(args.init_checkpoint) if args.init_checkpoint else None)

    # Assign separate learning rates to pretrained encoders and task-specific layers.
    pretrained_params = list(model.visual_encoder.parameters()) + list(model.text_encoder.parameters())
    fresh_params = list(model.fusion_module.parameters()) + list(model.classifier.parameters())
    if hasattr(model, 'aux_vis_head') and model.aux_vis_head is not None:
        fresh_params += list(model.aux_vis_head.parameters()) + list(model.aux_txt_head.parameters())

    optimizer = optim.AdamW([
        {'params': pretrained_params, 'lr': 5e-6},
        {'params': fresh_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay)
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    total_steps = math.ceil(len(train_loader) / args.accumulation_steps) * args.epochs
    warmup_steps = int(total_steps * 0.1)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    best_f1 = -1.0
    patience_counter = 0
    best_model_path = os.path.join(save_dir, 'best_model.pt')

    logger.info("Starting joint training...")

    for epoch in range(1, args.epochs + 1):

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion_ce, device, epoch,
            args.lambda_aux, args.accumulation_steps, scheduler, scaler
        )
        logger.info(f"[Epoch {epoch}] Train Total Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

        val_loss, val_metrics = evaluate(model, dev_loader, criterion_ce, device, args.lambda_aux, task_info['label_map'])
        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics[args.selection_metric]
        logger.info('Epoch %s | validation loss %.4f | metrics %s | selection %s', epoch, val_loss, val_metrics, args.selection_metric)



        # Select checkpoints using only the declared validation metric.
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved the best GMSNet checkpoint (validation {args.selection_metric}: {best_f1:.4f})")
        else:
            patience_counter += 1
            logger.info(f"No validation improvement. Early-stopping counter: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            logger.info("Early stopping criterion reached.")
            break

    logger.info("Training completed.")


if __name__ == "__main__":
    main()