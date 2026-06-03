# Experiment Log

This document records the experiments for the object detection assignment in `YeuCau.md`.
The goal is to build a self-contained detector for the five public classes:

- `person`
- `car`
- `dog`
- `cat`
- `chair`

Target metric: `mAP@0.5 >= 0.80` on the hidden test set if possible.

## Project Direction

We will keep the current custom YOLO/legacy detection network as the detection
pipeline, but replace the plain custom backbone with an ImageNet-pretrained
ResNet50 feature extractor.

The intended architecture is:

```text
Image
  -> JSON dataset loader and letterbox preprocessing
  -> ImageNet-pretrained ResNet50 backbone
  -> 1x1 channel adapters
  -> legacy YOLO FPN
  -> legacy YOLO detection head
  -> confidence threshold
  -> per-class NMS
  -> bbox scaling back to original image coordinates
  -> predictions.json
```

This keeps the detector implementation local and custom while satisfying the
backbone requirement with ImageNet-pretrained features.

## Constraints From Assignment

- Must implement dataset loading, preprocessing, augmentation, model, loss,
  inference, confidence thresholding, and NMS.
- Must not use complete object detection frameworks or complete pretrained
  detectors such as YOLOv5/v8, Detectron2, MMDetection, torchvision Faster
  R-CNN, or torchvision SSD.
- PyTorch and basic network layers are allowed.
- An ImageNet-pretrained classification backbone is allowed and required for
  our chosen direction.
- Required training command:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

- Required inference command:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```

- Required checkpoint output:

```text
./models/best.pth
```

## Dataset Notes

Current public split:

| Split | Images | Annotations |
| --- | ---: | ---: |
| Train | 7500 | 10642 |
| Val | 1500 | 2021 |

Class distribution:

| Class | Train | Val |
| --- | ---: | ---: |
| person | 5829 | 1074 |
| chair | 1613 | 282 |
| car | 1339 | 283 |
| dog | 1028 | 206 |
| cat | 833 | 176 |

The dataset is imbalanced toward `person`, so validation should be checked both
globally and per class. If weaker classes underperform, we should try class-aware
sampling, stronger augmentation, or lower per-class confidence thresholds.

## Planned Code Architecture

### `utils/json_dataset.py`

Purpose: load assignment JSON directly instead of relying on YOLO `.txt` labels.

Responsibilities:

- Read `train.json` and `val.json`.
- Resolve image paths from `image_dir` plus each image file name.
- Convert bbox from assignment format `[xmin, ymin, xmax, ymax]` to normalized
  training format `[cx, cy, w, h]`.
- Preserve multiple objects per image.
- Apply letterbox resize to the configured input size.
- Apply training augmentations:
  - horizontal flip
  - HSV/color jitter
  - optional random crop or mosaic, only if stable
- Return tensors in the same target structure expected by the existing loss:

```python
{
    "cls": class_ids,
    "box": normalized_xywh_boxes,
    "idx": batch_indices,
}
```

### `models/backbones.py`

Purpose: keep ImageNet-pretrained backbone choices isolated.

Selected backbone:

- `resnet50`: ImageNet-pretrained classification backbone from torchvision.
- We will use intermediate feature maps from `layer2`, `layer3`, and `layer4`.
- This is the primary experiment path because it is simple, stable, and easy to
  justify for the assignment.

Backbone candidates kept only for backup:

- `convnext_base`: stronger but heavier.
- `efficientnet_b0` or `efficientnet_b3`: useful if GPU memory is limited.

Each backbone should expose three feature maps at strides 8, 16, and 32:

```python
p3, p4, p5 = backbone(image)
```

### `models/detector.py`

Purpose: combine pretrained backbone with the custom legacy detector parts.

Initial detector:

```text
ResNet50 backbone
  -> adapter convs to legacy channel sizes
  -> legacy_nn.DarkFPN
  -> legacy_nn.Head(num_classes=5)
```

The adapters are important because pretrained backbones usually emit channels
such as `256/512/1024`, while the legacy FPN/head can be kept smaller.

### `train.py`

Purpose: assignment-compatible training entrypoint.

Responsibilities:

- Parse required CLI args.
- Build JSON train/val dataloaders.
- Build detector with ImageNet-pretrained backbone.
- Train with AMP when CUDA is available.
- Use existing YOLO-style loss from `utils/util.py` if compatible.
- Run validation every 5 epochs.
- Evaluate on val using assignment evaluator-compatible prediction format.
- Track validation `mAP@0.5` as the main model-selection metric.
- Save latest checkpoint to `models/last.pth` after each epoch.
- Save best checkpoint to `models/best.pth` whenever validation `mAP@0.5`
  improves at a validation epoch.
- Save a small run log to `models/train_log.json` or `models/train_log.csv`.

Checkpoint contents:

```python
{
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "classes": classes,
    "input_size": input_size,
    "backbone": "resnet50",
    "best_map50": best_map50,
}
```

Validation schedule:

- Run validation when `(epoch + 1) % 5 == 0`.
- Also run validation on the final epoch if it is not already a validation
  epoch.
- Save `models/last.pth` after every epoch, even when validation is skipped.

Validation outputs:

- `models/val_predictions.json`
- `models/val_score.json`
- `models/train_log.csv`

### `predict.py`

Purpose: assignment-compatible inference entrypoint.

Responsibilities:

- Parse `--image_dir`, `--output`, and optional checkpoint/config args.
- Load `models/best.pth` by default.
- Run inference for every image in `image_dir`.
- Apply confidence threshold and per-class NMS.
- Scale boxes back to original image coordinates.
- Write JSON array:

```json
[
  {
    "image_id": "img_7fd91a4c2e30.jpg",
    "boxes": [
      {
        "class": "person",
        "confidence": 0.91,
        "bbox": [48, 72, 210, 356]
      }
    ]
  }
]
```

Images without detections must still be included with `"boxes": []`.

## Experiment Template

Use this template for every real run.

```text
Experiment ID:
Date:
Code version / notes:

Backbone:
Pretrained weights:
Detector neck/head:
Input size:
Frozen layers:

Training:
Epochs:
Batch size:
Optimizer:
Learning rate:
Weight decay:
Augmentation:
Loss changes:

Inference:
Confidence threshold:
NMS IoU threshold:
Max detections:

Validation result:
mAP@0.5:
Precision:
Recall:
Per-class notes:

Outcome:
What improved:
What failed:
Next action:
```

## Experiment 000 - Baseline From Existing Repo

Status: reference only.

Backbone:

- Custom legacy DarkNet backbone.
- Not ImageNet-pretrained.

Detector:

- `legacy_nn.DarkNet`
- `legacy_nn.DarkFPN`
- `legacy_nn.Head`

Notes:

- Useful as a code baseline.
- Does not satisfy our chosen pretrained-backbone requirement.
- Keep this path as a sanity check for loss, NMS, and checkpoint loading.

Next action:

- Do not spend much tuning effort here.
- Move to ImageNet-pretrained backbone experiments.

## Experiment 001 - ConvNeXt-Base ImageNet Backbone + Legacy FPN/Head

Status: backup candidate.

Existing code:

- `nets/convnext_legacy.py`

Backbone:

- `torchvision.models.convnext_base`
- ImageNet weights: `ConvNeXt_Base_Weights.IMAGENET1K_V1`
- Feature maps:
  - stride 8: 256 channels
  - stride 16: 512 channels
  - stride 32: 1024 channels

Detector:

- 1x1 adapters:
  - `256 -> 128`
  - `512 -> 128`
  - `1024 -> 256`
- Legacy FPN:
  - `legacy_nn.DarkFPN`
- Legacy head:
  - `legacy_nn.Head(num_classes=5)`

Why this is plausible:

- ConvNeXt-Base has strong ImageNet features.
- The legacy detector head can remain custom.
- The adapter layer keeps the FPN/head small enough for faster training.

Risks:

- ConvNeXt-Base may be heavy for local GPU memory.
- Existing loss expects YOLO-style training outputs, so target formatting must
  match exactly.
- If backbone is fully trainable from epoch 1, training may overfit or become
  unstable on 7500 images.

Initial training plan:

- Input size: `640`
- Backbone freeze: freeze first 3 to 5 epochs, then unfreeze.
- Batch size: largest stable CUDA batch; fallback to gradient accumulation.
- Optimizer: SGD or AdamW; start with AdamW for pretrained backbone fine-tuning.
- Learning rate:
  - backbone: lower LR
  - FPN/head/adapters: higher LR
- Augmentation:
  - horizontal flip
  - HSV jitter
  - mild scale/translate
  - avoid aggressive mosaic until baseline is stable

Initial inference plan:

- Confidence threshold: `0.25`
- NMS IoU threshold: `0.50`
- Evaluate thresholds from `0.10` to `0.50` after first usable checkpoint.

Decision rule:

- Use only if ResNet50 cannot reach a competitive validation score.

## Experiment 002 - ResNet50 ImageNet Backbone + Legacy FPN/Head

Status: selected primary experiment.

Backbone:

- `torchvision.models.resnet50`
- ImageNet weights: `ResNet50_Weights.IMAGENET1K_V2`
- Feature maps:
  - `layer2`: stride 8, 512 channels
  - `layer3`: stride 16, 1024 channels
  - `layer4`: stride 32, 2048 channels

Detector:

- 1x1 adapters:
  - `512 -> 128`
  - `1024 -> 128`
  - `2048 -> 256`
- Legacy FPN:
  - `legacy_nn.DarkFPN`
- Legacy head:
  - `legacy_nn.Head(num_classes=5)`

Why this is plausible:

- Easier to explain in the final README/report.
- Usually lighter and more predictable than ConvNeXt-Base.
- Uses a standard ImageNet-pretrained classification backbone.
- Keeps object detection layers custom, which matches the assignment rules.

Risks:

- May underperform ConvNeXt on small objects and cluttered scenes.
- Requires careful channel adapters because output channels are larger.

Initial training plan:

- Input size: `640`
- Warmup freeze: freeze the full ResNet50 backbone for the first 5 epochs.
- After warmup: unfreeze only `layer4` for light fine-tuning.
- Keep `conv1`, `bn1`, `layer1`, `layer2`, and `layer3` frozen unless the
  validation score clearly plateaus.
- Train adapters, legacy FPN, and legacy head from the beginning.
- Optimizer: AdamW for the first version.
- Learning rate:
  - ResNet50 `layer4`: small LR, around `0.1x` the head LR
  - adapters/FPN/head: larger LR
- Augmentation:
  - horizontal flip
  - HSV/color jitter
  - mild scale/translate
  - delay mosaic until the baseline is stable

Fine-tuning schedule:

- Epochs `1-5`: train adapters, legacy FPN, and legacy head only.
- Epoch `6+`: unfreeze ResNet50 `layer4` and continue training with a lower LR
  for `layer4`.
- If validation becomes unstable after unfreezing, freeze `layer4` again and
  continue training only adapters/FPN/head.

Augmentation policy:

- Start with conservative augmentations because the backbone is pretrained and
  the dataset is medium-sized.
- Always use letterbox resize so bbox geometry stays consistent.
- Enable horizontal flip because all five classes are horizontally symmetric
  enough for this dataset.
- Use mild HSV/color jitter to improve robustness to lighting and camera style.
- Use mild scale and translate to improve localization robustness.
- Avoid vertical flip because it creates unrealistic images for people, cars,
  chairs, cats, and dogs.
- Avoid heavy random crop at first because it can cut away small objects or
  make labels noisy.
- Treat mosaic as an ablation after the first stable ResNet50 run; it may help
  small-object recall, but it can also make pretrained backbone fine-tuning less
  stable.

Initial inference plan:

- Confidence threshold: `0.25`
- NMS IoU threshold: `0.50`
- Sweep confidence threshold after the first trained checkpoint.

Success criteria:

- `predict.py` output passes the public evaluator format checks.
- Validation `mAP@0.5` improves over the custom non-pretrained baseline.
- Recall remains acceptable for lower-count classes: `cat`, `dog`, and `car`.

## Experiment 003 - EfficientNet ImageNet Backbone + Legacy FPN/Head

Status: optional candidate.

Backbone:

- `efficientnet_b0` for low memory.
- `efficientnet_b3` if GPU memory allows.

Why this is plausible:

- EfficientNet can be strong with fewer parameters.
- Useful if we need faster iteration.

Risks:

- Extracting clean stride 8/16/32 features is less straightforward than ResNet.
- May need more careful normalization and feature adapter design.

Decision rule:

- Try only after ConvNeXt and ResNet50 are evaluated.

## Ablation Plan

Backbone ablations:

- ResNet50 frozen warmup vs fully trainable from start.
- ResNet50 adapter width: small legacy channels vs larger channels.
- ResNet50 vs ConvNeXt-Base only if the selected path plateaus.

Training ablations:

- Input size `640` vs `768`.
- AdamW vs SGD.
- Mild augmentation vs mosaic enabled.
- Class-aware sampling for underrepresented classes.

Inference ablations:

- Confidence threshold sweep: `0.10`, `0.15`, `0.20`, `0.25`, `0.30`, `0.40`.
- NMS IoU sweep: `0.45`, `0.50`, `0.55`, `0.60`, `0.65`.
- Class-specific thresholds if `person` dominates false positives.

## Current Next Steps

1. Add `utils/json_dataset.py`.
2. Add `models/backbones.py` with a ResNet50 feature extractor.
3. Add `models/detector.py` with ResNet50 adapters plus legacy FPN/head.
4. Add assignment-compatible `train.py`.
5. Add assignment-compatible `predict.py`.
6. Run a smoke test on a tiny train subset.
7. Run full validation prediction and evaluate with:

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions val_predictions.json \
  --output val_score.json
```

8. Log every result in this file before changing another variable.
