import os
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
from sklearn.metrics import accuracy_score, f1_score
from transformers import get_cosine_schedule_with_warmup

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
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps (effective batch size = batch_size * steps)')

    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--patience', type=int, default=4)

    parser.add_argument('--lambda_aux', type=float, default=0.05, help='Weight applied to the sum of the auxiliary classification losses')

    parser.add_argument('--visual_weights', type=str, default='./local_models/resnet50-0676ba61.pth')
    parser.add_argument('--text_model_path', type=str, default='../local_models/deberta-v3-base')
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.4)

    return parser.parse_args()



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
    total_loss, total_ce, total_cl = 0, 0, 0
    correct, total = 0, 0

    optimizer.zero_grad()

    loop = tqdm(loader, desc=f"Train Epoch {epoch}")
    for i, batch in enumerate(loop):
        images = batch['image'].to(device)
        text_inputs = {k: v.to(device) for k, v in batch['text_tokens'].items()}
        labels = batch['label'].to(device)

        inputs = {'image': images, 'text_tokens': text_inputs}

        with torch.amp.autocast('cuda'):
            logits, aux_v_logits, aux_t_logits = model(inputs, return_features=True)

            loss_ce = criterion_ce(logits, labels)

            loss_aux_v = criterion_ce(aux_v_logits, labels)
            loss_aux_t = criterion_ce(aux_t_logits, labels)

            loss_aux = loss_aux_v + loss_aux_t

            # Combine the main classification loss with unimodal auxiliary losses.
            loss = loss_ce + lambda_aux * loss_aux
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        # Update parameters after accumulation, including the final partial group.
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        total_ce += loss_ce.item()
        total_cl += loss_aux.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(
            loss=loss.item() * accumulation_steps,
            ce=loss_ce.item(),
            aux=loss_aux.item(),
            acc=correct / total
        )

    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion_ce, device, lambda_aux):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            text_inputs = {k: v.to(device) for k, v in batch['text_tokens'].items()}
            labels = batch['label'].to(device)

            inputs = {'image': images, 'text_tokens': text_inputs}

            logits, aux_v_logits, aux_t_logits = model(inputs, return_features=True)

            loss_ce = criterion_ce(logits, labels)
            loss_aux_v = criterion_ce(aux_v_logits, labels)
            loss_aux_t = criterion_ce(aux_t_logits, labels)

            loss_aux = loss_aux_v + loss_aux_t
            loss = loss_ce + lambda_aux * loss_aux

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(loader), acc, f1





def main():
    args = parse_args()
    set_seed(args.seed)

    save_dir = os.path.join(args.output_dir, args.task_name, args.run_name)
    logger = setup_logger(save_dir)
    logger.info(f"Starting GMSNet training: {args.run_name}")

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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
    total_steps = len(train_loader) // args.accumulation_steps * args.epochs
    warmup_steps = int(total_steps * 0.1)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler()

    best_f1 = 0.0
    patience_counter = 0
    best_model_path = os.path.join(save_dir, 'best_model.pt')

    if os.path.exists(best_model_path):
        logger.info("\n" + "=" * 40)
        logger.info(f"Found existing best model: {best_model_path}")
        logger.info("Loading model weights; optimizer and scheduler states are not restored...")
        try:
            state_dict = torch.load(best_model_path, map_location=device)
            model.load_state_dict(state_dict)
            logger.info("Model weights loaded.")

            logger.info("Evaluating the loaded checkpoint on the validation split...")
            _, _, init_f1 = evaluate(model, dev_loader, criterion_ce, device, args.lambda_aux)
            best_f1 = init_f1
            logger.info(f"Initial validation weighted F1: {best_f1:.4f}")
            logger.info("=" * 40 + "\n")

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.info("Continuing with the current model parameters after checkpoint loading failed.")
            logger.info("=" * 40 + "\n")
    else:
        logger.info("No existing checkpoint found. Starting a new training run.")

    logger.info("Starting joint training...")

    for epoch in range(1, args.epochs + 1):

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion_ce, device, epoch,
            args.lambda_aux, args.accumulation_steps, scheduler, scaler
        )
        logger.info(f"[Epoch {epoch}] Train Total Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

        val_loss, val_acc, val_f1 = evaluate(model, dev_loader, criterion_ce, device, args.lambda_aux)
        logger.info(f"[Epoch {epoch}] Val Total Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Weighted F1: {val_f1:.4f}")



        # Select checkpoints and apply early stopping using validation weighted F1.
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved the best GMSNet checkpoint (validation weighted F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            logger.info(f"No validation improvement. Early-stopping counter: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            logger.info("Early stopping criterion reached.")
            break

    logger.info("Training completed.")


if __name__ == "__main__":
    main()