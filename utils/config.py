from pathlib import Path

import yaml


DEFAULT_CONFIG = {
    'model': {
        'backbone': 'resnet50',
        'pretrained_backbone': True,
        'input_size': 640,
    },
    'training': {
        'epochs': 50,
        'batch_size': 8,
        'num_workers': 4,
        'lr': 1e-3,
        'backbone_lr': 1e-4,
        'layer3_lr': 2e-5,
        'weight_decay': 5e-4,
        'warmup_epochs': 5,
        'warmup_start_factor': 0.1,
        'scheduler': 'constant',
        'min_lr': 1e-5,
        'min_backbone_lr': 1e-6,
        'min_layer3_lr': 1e-6,
        'layer4_unfreeze_epoch': 6,
        'layer3_unfreeze_epoch': 0,
        'val_interval': 5,
        'val_every_epoch_from': 0,
        'early_stopping_patience': 0,
        'ema': {
            'enabled': False,
            'decay': 0.9999,
            'tau': 2000,
        },
        'sampling': {
            'strategy': 'uniform',
            'num_samples': 0,
            'empty_fraction': 0.20,
            'class_weights': {},
            'seed': 0,
        },
    },
    'inference': {
        'confidence': 0.25,
        'iou': 0.50,
    },
    'loss': {
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
    },
    'augmentation': {
        'flip_lr': 0.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'translate': 0.10,
        'scale': 0.20,
    },
}


def deep_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path='config/current_best.yaml'):
    config = deep_update({}, DEFAULT_CONFIG)
    path = Path(config_path)
    if path.exists():
        with path.open('r', encoding='utf-8') as file:
            loaded = yaml.safe_load(file) or {}
        deep_update(config, loaded)
    return config


def flatten_training_params(config):
    params = {}
    params.update(config.get('loss', {}))
    params.update(config.get('augmentation', {}))
    return params
