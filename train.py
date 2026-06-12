import argparse
import csv
import json
import math
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
    parser.add_argument('--config', default='config/official.yaml')
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
    parser.add_argument('--layer3_lr', default=None, type=float)
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
    args.layer3_lr = args.layer3_lr or config['training']['layer3_lr']
    args.weight_decay = args.weight_decay or config['training']['weight_decay']
    args.warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else config['training']['warmup_epochs']
    args.val_interval = args.val_interval if args.val_interval is not None else config['training']['val_interval']
    args.confidence = args.confidence if args.confidence is not None else config['inference']['confidence']
    args.iou = args.iou if args.iou is not None else config['inference']['iou']

    training = config['training']
    args.scheduler = training.get('scheduler', 'constant').lower()
    args.warmup_start_factor = training.get('warmup_start_factor', 0.1)
    args.min_lr = training.get('min_lr', args.lr)
    args.min_backbone_lr = training.get('min_backbone_lr', args.backbone_lr)
    args.min_layer3_lr = training.get('min_layer3_lr', args.layer3_lr)
    args.layer4_unfreeze_epoch = training.get('layer4_unfreeze_epoch', args.warmup_epochs + 1)
    args.layer3_unfreeze_epoch = training.get('layer3_unfreeze_epoch', 0)
    args.val_every_epoch_from = training.get('val_every_epoch_from', 0)
    args.early_stopping_patience = training.get('early_stopping_patience', 0)

    ema_config = training.get('ema', {})
    args.ema_enabled = bool(ema_config.get('enabled', False))
    args.ema_decay = float(ema_config.get('decay', 0.9999))
    args.ema_tau = float(ema_config.get('tau', 2000))
    return args


def configure_backbone_for_epoch(model, epoch, layer4_unfreeze_epoch, layer3_unfreeze_epoch):
    epoch_number = epoch + 1
    model.freeze_backbone()
    finetune_layer4 = layer4_unfreeze_epoch > 0 and epoch_number >= layer4_unfreeze_epoch
    finetune_layer3 = layer3_unfreeze_epoch > 0 and epoch_number >= layer3_unfreeze_epoch

    if finetune_layer4:
        model.unfreeze_layer4()
    if finetune_layer3:
        model.unfreeze_layer3()

    model.set_backbone_train_mode()
    if finetune_layer3:
        stage = 'layer3+layer4'
    elif finetune_layer4:
        stage = 'layer4'
    else:
        stage = 'frozen'
    return {
        'stage': stage,
        'layer3_finetuned': finetune_layer3,
        'layer4_finetuned': finetune_layer4,
    }


def build_optimizer(model, args):
    layer3_params = list(model.backbone.layer3.parameters())
    layer4_params = list(model.backbone.layer4.parameters())
    backbone_param_ids = {
        id(parameter)
        for parameter in list(model.backbone.parameters())
    }
    head_params = [
        parameter for parameter in model.parameters()
        if id(parameter) not in backbone_param_ids
    ]
    return torch.optim.AdamW([
        {
            'params': head_params,
            'lr': args.lr,
            'name': 'head',
            'peak_lr': args.lr,
            'min_lr': args.min_lr,
        },
        {
            'params': layer4_params,
            'lr': args.backbone_lr,
            'name': 'layer4',
            'peak_lr': args.backbone_lr,
            'min_lr': args.min_backbone_lr,
        },
        {
            'params': layer3_params,
            'lr': args.layer3_lr,
            'name': 'layer3',
            'peak_lr': args.layer3_lr,
            'min_lr': args.min_layer3_lr,
        },
    ], weight_decay=args.weight_decay)


class WarmupCosineScheduler:
    def __init__(self, optimizer, total_steps, warmup_steps, start_factor=0.1):
        self.optimizer = optimizer
        self.total_steps = max(int(total_steps), 1)
        self.warmup_steps = min(max(int(warmup_steps), 0), self.total_steps)
        self.start_factor = float(start_factor)
        self.last_step = -1

    def get_lr(self, group, step):
        peak_lr = float(group['peak_lr'])
        min_lr = float(group['min_lr'])

        if self.warmup_steps and step < self.warmup_steps:
            progress = (step + 1) / self.warmup_steps
            return peak_lr * (
                self.start_factor + (1.0 - self.start_factor) * progress
            )

        decay_steps = max(self.total_steps - self.warmup_steps - 1, 1)
        decay_step = min(max(step - self.warmup_steps, 0), decay_steps)
        progress = decay_step / decay_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (peak_lr - min_lr) * cosine

    def step(self, step):
        self.last_step = int(step)
        for group in self.optimizer.param_groups:
            group['lr'] = self.get_lr(group, self.last_step)

    def state_dict(self):
        return {
            'total_steps': self.total_steps,
            'warmup_steps': self.warmup_steps,
            'start_factor': self.start_factor,
            'last_step': self.last_step,
        }


def build_scheduler(optimizer, args, steps_per_epoch):
    if args.scheduler == 'constant':
        return None
    if args.scheduler != 'cosine':
        raise ValueError(f'Unsupported scheduler: {args.scheduler}')
    return WarmupCosineScheduler(
        optimizer=optimizer,
        total_steps=args.epochs * steps_per_epoch,
        warmup_steps=args.warmup_epochs * steps_per_epoch,
        start_factor=args.warmup_start_factor,
    )


def optimizer_lrs(optimizer):
    return {
        group.get('name', str(index)): group['lr']
        for index, group in enumerate(optimizer.param_groups)
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    ema,
    epoch,
    classes,
    input_size,
    best_map50,
    config,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_model = ema.ema if ema is not None else model
    torch.save({
        'epoch': epoch,
        'model': checkpoint_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': None if scheduler is None else scheduler.state_dict(),
        'ema_updates': 0 if ema is None else ema.updates,
        'uses_ema_weights': ema is not None,
        'classes': classes,
        'input_size': input_size,
        'backbone': 'resnet50',
        'best_map50': best_map50,
        'config': config,
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
    score_path.write_text(
        json.dumps(score, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return score


def preserve_best_validation_artifacts(checkpoint_dir):
    for source_name, target_name in (
        ('val_predictions.json', 'best_val_predictions.json'),
        ('val_score.json', 'best_val_score.json'),
    ):
        source = checkpoint_dir / source_name
        target = checkpoint_dir / target_name
        target.write_bytes(source.read_bytes())


def append_log(log_path, row):
    exists = log_path.exists()
    with log_path.open('a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'epoch', 'train_loss', 'box_loss', 'cls_loss', 'dfl_loss',
            'map50', 'precision', 'recall', 'it_per_second', 'epoch_seconds',
            'backbone_stage', 'layer3_finetuned', 'layer4_finetuned',
            'lr_head', 'lr_layer4', 'lr_layer3', 'ema_updates',
        ])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, ema, scaler, args, device):
    model.train()
    backbone_state = configure_backbone_for_epoch(
        model,
        args.current_epoch,
        args.layer4_unfreeze_epoch,
        args.layer3_unfreeze_epoch,
    )

    total_loss = 0.0
    box_loss_total = 0.0
    cls_loss_total = 0.0
    dfl_loss_total = 0.0
    num_batches = 0
    start_time = time.time()

    optimizer.zero_grad(set_to_none=True)
    for batch_index, (samples, targets) in enumerate(loader):
        if scheduler is not None:
            global_step = args.current_epoch * len(loader) + batch_index
            scheduler.step(global_step)

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
        if ema is not None:
            ema.update(model)

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
        **backbone_state,
    }


def should_validate_epoch(epoch_number, args):
    validate_by_interval = epoch_number % args.val_interval == 0
    validate_every_epoch = (
        args.val_every_epoch_from > 0
        and epoch_number >= args.val_every_epoch_from
    )
    return validate_by_interval or validate_every_epoch or epoch_number == args.epochs


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
    scheduler = build_scheduler(optimizer, args, len(train_loader))
    criterion = util.ComputeLoss(model, params)
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    ema = (
        util.EMA(model, decay=args.ema_decay, tau=args.ema_tau)
        if args.ema_enabled
        else None
    )

    best_map50 = 0.0
    epochs_without_improvement = 0
    log_path = checkpoint_dir / 'train_log.csv'

    for epoch in range(args.epochs):
        args.current_epoch = epoch
        metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            ema,
            scaler,
            args,
            device,
        )

        epoch_number = epoch + 1
        should_validate = should_validate_epoch(epoch_number, args)
        val_score = None
        improved = False
        if should_validate:
            validation_model = ema.ema if ema is not None else model
            val_score = evaluate_validation(
                validation_model,
                args,
                classes,
                checkpoint_dir,
                epoch,
            )
            map50 = float(val_score['mAP@0.5'])
            if map50 > best_map50:
                best_map50 = map50
                improved = True
                epochs_without_improvement = 0
                preserve_best_validation_artifacts(checkpoint_dir)
                save_checkpoint(
                    checkpoint_dir / 'best.pth',
                    model,
                    optimizer,
                    scheduler,
                    ema,
                    epoch_number,
                    classes,
                    args.input_size,
                    best_map50,
                    config,
                )
            else:
                epochs_without_improvement += 1
        else:
            map50 = ''

        save_checkpoint(
            checkpoint_dir / 'last.pth',
            model,
            optimizer,
            scheduler,
            ema,
            epoch_number,
            classes,
            args.input_size,
            best_map50,
            config,
        )

        learning_rates = optimizer_lrs(optimizer)
        append_log(log_path, {
            'epoch': epoch_number,
            'train_loss': round(metrics['loss'], 6),
            'box_loss': round(metrics['box'], 6),
            'cls_loss': round(metrics['cls'], 6),
            'dfl_loss': round(metrics['dfl'], 6),
            'map50': map50,
            'precision': '' if val_score is None else val_score['micro_precision'],
            'recall': '' if val_score is None else val_score['micro_recall'],
            'it_per_second': round(metrics['it_per_second'], 4),
            'epoch_seconds': round(metrics['epoch_seconds'], 2),
            'backbone_stage': metrics['stage'],
            'layer3_finetuned': metrics['layer3_finetuned'],
            'layer4_finetuned': metrics['layer4_finetuned'],
            'lr_head': learning_rates['head'],
            'lr_layer4': learning_rates['layer4'],
            'lr_layer3': learning_rates['layer3'],
            'ema_updates': 0 if ema is None else ema.updates,
        })

        print(
            f"epoch {epoch_number}/{args.epochs} "
            f"loss={metrics['loss']:.4f} "
            f"it/s={metrics['it_per_second']:.2f} "
            f"epoch_time={metrics['epoch_seconds']:.1f}s "
            f"stage={metrics['stage']} "
            f"lr={learning_rates['head']:.2e}/"
            f"{learning_rates['layer4']:.2e}/"
            f"{learning_rates['layer3']:.2e} "
            f"map50={map50 if map50 != '' else 'skip'} "
            f"best={best_map50:.4f}"
        )

        if (
            should_validate
            and not improved
            and args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f'early stopping at epoch {epoch_number}: '
                f'no validation improvement for '
                f'{epochs_without_improvement} validation epochs'
            )
            break

    if best_map50 == 0.0 and not (checkpoint_dir / 'best.pth').exists():
        save_checkpoint(
            checkpoint_dir / 'best.pth',
            model,
            optimizer,
            scheduler,
            ema,
            epoch_number,
            classes,
            args.input_size,
            best_map50,
            config,
        )


if __name__ == '__main__':
    main()
