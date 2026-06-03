import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils import data

from models import ResNet50LegacyYOLO
from predict import run_predictions
from utils import util
from utils.config import flatten_training_params, load_config
from utils.json_dataset import JsonDetectionDataset, load_annotations


def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet50 legacy YOLO detector.')
    parser.add_argument('--config', default='config/hyperparameters.yaml')
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--val_data', required=True)
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--val_image_dir', required=True)
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--input_size', default=None, type=int)
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--num_workers', default=None, type=int)
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--backbone_lr', default=None, type=float)
    parser.add_argument('--weight_decay', default=None, type=float)
    parser.add_argument('--warmup_epochs', default=None, type=int)
    parser.add_argument('--val_interval', default=None, type=int)
    parser.add_argument('--confidence', default=None, type=float)
    parser.add_argument('--iou', default=None, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no_pretrained', action='store_true')
    return parser.parse_args()


def apply_config(args, config):
    args.input_size = args.input_size or config['model']['input_size']
    args.epochs = args.epochs or config['training']['epochs']
    args.batch_size = args.batch_size or config['training']['batch_size']
    args.num_workers = args.num_workers if args.num_workers is not None else config['training']['num_workers']
    args.lr = args.lr or config['training']['lr']
    args.backbone_lr = args.backbone_lr or config['training']['backbone_lr']
    args.weight_decay = args.weight_decay or config['training']['weight_decay']
    args.warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else config['training']['warmup_epochs']
    args.val_interval = args.val_interval if args.val_interval is not None else config['training']['val_interval']
    args.confidence = args.confidence if args.confidence is not None else config['inference']['confidence']
    args.iou = args.iou if args.iou is not None else config['inference']['iou']
    return args


def configure_backbone_for_epoch(model, epoch, warmup_epochs):
    model.freeze_backbone()
    finetune_layer4 = epoch + 1 > warmup_epochs
    if finetune_layer4:
        model.unfreeze_layer4()
    model.set_backbone_train_mode(finetune_layer4=finetune_layer4)
    return finetune_layer4


def build_optimizer(model, args):
    layer4_params = list(model.backbone.layer4.parameters())
    layer4_param_ids = {id(parameter) for parameter in layer4_params}
    head_params = [
        parameter for parameter in model.parameters()
        if id(parameter) not in layer4_param_ids and parameter.requires_grad
    ]
    return torch.optim.AdamW([
        {'params': head_params, 'lr': args.lr},
        {'params': layer4_params, 'lr': args.backbone_lr},
    ], weight_decay=args.weight_decay)


def save_checkpoint(path, model, optimizer, epoch, classes, input_size, best_map50):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'classes': classes,
        'input_size': input_size,
        'backbone': 'resnet50',
        'best_map50': best_map50,
    }, path)


def evaluate_validation(model, args, classes, checkpoint_dir, epoch):
    prediction_path = checkpoint_dir / 'val_predictions.json'
    score_path = checkpoint_dir / 'val_score.json'

    model.eval()
    predictions = run_predictions(
        model=model,
        image_dir=args.val_image_dir,
        classes=classes,
        input_size=args.input_size,
        confidence=args.confidence,
        iou=args.iou,
        device=torch.device(args.device),
    )
    prediction_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    subprocess.run([
        sys.executable,
        'public/tools/evaluate_predictions.py',
        '--ground_truth',
        args.val_data,
        '--predictions',
        str(prediction_path),
        '--output',
        str(score_path),
    ], check=True)

    score = json.loads(score_path.read_text(encoding='utf-8'))
    score['epoch'] = epoch + 1
    return score


def append_log(log_path, row):
    exists = log_path.exists()
    with log_path.open('a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'epoch', 'train_loss', 'box_loss', 'cls_loss', 'dfl_loss',
            'map50', 'precision', 'recall', 'it_per_second', 'epoch_seconds',
            'layer4_finetuned',
        ])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_one_epoch(model, loader, criterion, optimizer, scaler, args, device):
    model.train()
    layer4_finetuned = configure_backbone_for_epoch(model, args.current_epoch, args.warmup_epochs)

    total_loss = 0.0
    box_loss_total = 0.0
    cls_loss_total = 0.0
    dfl_loss_total = 0.0
    num_batches = 0
    start_time = time.time()

    optimizer.zero_grad(set_to_none=True)
    for samples, targets in loader:
        samples = samples.to(device).float() / 255.0
        targets = {key: value.to(device) for key, value in targets.items()}

        with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            outputs = model(samples)
            loss_box, loss_cls, loss_dfl = criterion(outputs, targets)
            loss = loss_box + loss_cls + loss_dfl

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().cpu())
        box_loss_total += float(loss_box.detach().cpu())
        cls_loss_total += float(loss_cls.detach().cpu())
        dfl_loss_total += float(loss_dfl.detach().cpu())
        num_batches += 1

    divisor = max(num_batches, 1)
    elapsed = max(time.time() - start_time, 1e-9)
    return {
        'loss': total_loss / divisor,
        'box': box_loss_total / divisor,
        'cls': cls_loss_total / divisor,
        'dfl': dfl_loss_total / divisor,
        'it_per_second': num_batches / elapsed,
        'epoch_seconds': elapsed,
        'layer4_finetuned': layer4_finetuned,
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    args = apply_config(args, config)
    device = torch.device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    params = flatten_training_params(config)
    _, classes, _ = load_annotations(args.train_data)

    train_dataset = JsonDetectionDataset(
        args.train_data,
        args.image_dir,
        input_size=args.input_size,
        augment=True,
        params=params,
    )
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        collate_fn=JsonDetectionDataset.collate_fn,
    )

    model = ResNet50LegacyYOLO(
        num_classes=len(classes),
        pretrained_backbone=(not args.no_pretrained and config['model'].get('pretrained_backbone', True)),
    ).to(device)
    model.freeze_backbone()

    optimizer = build_optimizer(model, args)
    criterion = util.ComputeLoss(model, params)
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')

    best_map50 = 0.0
    log_path = checkpoint_dir / 'train_log.csv'

    for epoch in range(args.epochs):
        args.current_epoch = epoch
        metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, args, device)

        save_checkpoint(
            checkpoint_dir / 'last.pth',
            model,
            optimizer,
            epoch + 1,
            classes,
            args.input_size,
            best_map50,
        )

        should_validate = (epoch + 1) % args.val_interval == 0 or epoch + 1 == args.epochs
        val_score = None
        if should_validate:
            val_score = evaluate_validation(model, args, classes, checkpoint_dir, epoch)
            map50 = float(val_score['mAP@0.5'])
            if map50 > best_map50:
                best_map50 = map50
                save_checkpoint(
                    checkpoint_dir / 'best.pth',
                    model,
                    optimizer,
                    epoch + 1,
                    classes,
                    args.input_size,
                    best_map50,
                )
        else:
            map50 = ''

        append_log(log_path, {
            'epoch': epoch + 1,
            'train_loss': round(metrics['loss'], 6),
            'box_loss': round(metrics['box'], 6),
            'cls_loss': round(metrics['cls'], 6),
            'dfl_loss': round(metrics['dfl'], 6),
            'map50': map50,
            'precision': '' if val_score is None else val_score['micro_precision'],
            'recall': '' if val_score is None else val_score['micro_recall'],
            'it_per_second': round(metrics['it_per_second'], 4),
            'epoch_seconds': round(metrics['epoch_seconds'], 2),
            'layer4_finetuned': metrics['layer4_finetuned'],
        })

        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"loss={metrics['loss']:.4f} "
            f"it/s={metrics['it_per_second']:.2f} "
            f"epoch_time={metrics['epoch_seconds']:.1f}s "
            f"map50={map50 if map50 != '' else 'skip'} "
            f"best={best_map50:.4f}"
        )

    if best_map50 == 0.0 and not (checkpoint_dir / 'best.pth').exists():
        save_checkpoint(
            checkpoint_dir / 'best.pth',
            model,
            optimizer,
            args.epochs,
            classes,
            args.input_size,
            best_map50,
        )


if __name__ == '__main__':
    main()
