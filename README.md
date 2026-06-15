# Object Detection

This project implements a ResNet50 + FPN object detector, including training,
inference, NMS, and consensus ensemble.

## 1. Run Locally

Run the following commands from the project root directory.

### Installation

Python 3.10 or newer and a virtual environment are recommended:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA support, install the PyTorch version compatible with your CUDA
installation and GPU driver by following the official PyTorch instructions.

### Local Prediction

The two-model ensemble is enabled by default:

```powershell
New-Item -ItemType Directory -Force grading_outputs | Out-Null

python predict.py `
  --image_dir public/val/images `
  --output grading_outputs/val_predictions.json
```

`predict.py` uses:

- `models/best.pth`
- `models/best2.pth`

If either checkpoint is missing, it is downloaded automatically from the
`minhdang0901/yolo-detector` repository on Hugging Face.

To run only `best.pth`:

```powershell
python predict.py `
  --image_dir public/val/images `
  --output grading_outputs/val_predictions.json `
  --single_model
```

Evaluate the predictions:

```powershell
python public/tools/evaluate_predictions.py `
  --ground_truth public/annotations/val.json `
  --predictions grading_outputs/val_predictions.json `
  --output grading_outputs/val_score.json
```

### Local Training

```powershell
python train.py `
  --config utils/hyperparameters.yaml `
  --train_data public/annotations/train.json `
  --val_data public/annotations/val.json `
  --image_dir public/train/images `
  --val_image_dir public/val/images `
  --checkpoint_dir models/training_output
```

The default training parameters are loaded from
`utils/hyperparameters.yaml`. Checkpoints, training logs, and validation
results are written to the directory passed through `--checkpoint_dir`.

## 2. Run with Docker

Docker Desktop must be running. Build the image from the project root:

```powershell
docker build -t object-detection-exam:2026 .
```

### Docker Prediction

```powershell
New-Item -ItemType Directory -Force grading_outputs | Out-Null

docker run --rm --gpus all `
  -v "${PWD}:/workspace" `
  -v "${PWD}/public/val/images:/exam/val_images:ro" `
  -v "${PWD}/grading_outputs:/exam/outputs" `
  object-detection-exam:2026 `
  python predict.py `
    --image_dir /exam/val_images `
    --output /exam/outputs/val_predictions.json
```

If no NVIDIA GPU is available, remove `--gpus all` and add `--device cpu` to
the `predict.py` command.

Evaluate the predictions on the host machine:

```powershell
python public/tools/evaluate_predictions.py `
  --ground_truth public/annotations/val.json `
  --predictions grading_outputs/val_predictions.json `
  --output grading_outputs/val_score.json
```

### Docker Training

```powershell
New-Item -ItemType Directory -Force training_outputs | Out-Null

docker run --rm --gpus all `
  -v "${PWD}:/workspace" `
  -v "${PWD}/training_outputs:/exam/training_outputs" `
  object-detection-exam:2026 `
  python train.py `
    --config utils/hyperparameters.yaml `
    --train_data public/annotations/train.json `
    --val_data public/annotations/val.json `
    --image_dir public/train/images `
    --val_image_dir public/val/images `
    --checkpoint_dir /exam/training_outputs
```

The checkpoints and training logs are saved in `training_outputs` on the host
machine. If no NVIDIA GPU is available, remove `--gpus all`, add
`--device cpu`, and consider reducing `batch_size` in
`utils/hyperparameters.yaml`.
