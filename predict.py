import argparse
import json
from pathlib import Path

import cv2
import torch

from models import ResNet50LegacyYOLO
from utils import util
from utils.config import load_config
from utils.json_dataset import letterbox, list_image_files, scale_boxes_to_original


def parse_args():
    parser = argparse.ArgumentParser(description='Run object detection inference.')
    parser.add_argument('--config', default='config/hyperparameters.yaml')
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', default='./models/best.pth')
    parser.add_argument('--input_size', default=None, type=int)
    parser.add_argument('--confidence', default=None, type=float)
    parser.add_argument('--iou', default=None, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint.get('classes', ['person', 'car', 'dog', 'cat', 'chair'])
    model = ResNet50LegacyYOLO(num_classes=len(classes), pretrained_backbone=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    return model, classes, checkpoint.get('input_size', 640)


def predict_image(model, image_path, classes, input_size, confidence, iou, device):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f'Unable to read image: {image_path}')

    height, width = image.shape[:2]
    resized, ratio, pad = letterbox(image, input_size)
    tensor = torch.from_numpy(resized.transpose((2, 0, 1))[::-1].copy())
    tensor = tensor.unsqueeze(0).to(device).float() / 255.0

    with torch.no_grad():
        outputs = model(tensor)
        detections = util.non_max_suppression(outputs, confidence, iou)[0]

    boxes = []
    if detections.shape[0]:
        detections = detections.detach().cpu()
        detections[:, :4] = scale_boxes_to_original(detections[:, :4], ratio, pad, width, height)
        for det in detections.tolist():
            x1, y1, x2, y2, conf, cls_id = det
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append({
                'class': classes[int(cls_id)],
                'confidence': round(float(conf), 6),
                'bbox': [
                    round(float(x1), 3),
                    round(float(y1), 3),
                    round(float(x2), 3),
                    round(float(y2), 3),
                ],
            })

    return {'image_id': image_path.name, 'boxes': boxes}


def run_predictions(model, image_dir, classes, input_size, confidence, iou, device):
    predictions = []
    for image_path in list_image_files(image_dir):
        predictions.append(predict_image(model, image_path, classes, input_size, confidence, iou, device))
    return predictions


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    model, classes, checkpoint_input_size = load_model(args.checkpoint, device)
    input_size = args.input_size or checkpoint_input_size or config['model']['input_size']
    confidence = args.confidence if args.confidence is not None else config['inference']['confidence']
    iou = args.iou if args.iou is not None else config['inference']['iou']
    predictions = run_predictions(
        model=model,
        image_dir=args.image_dir,
        classes=classes,
        input_size=input_size,
        confidence=confidence,
        iou=iou,
        device=device,
    )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'saved predictions to {output_path}')


if __name__ == '__main__':
    main()
