import argparse
import json
import math
from pathlib import Path

import cv2
import torch

from models import ResNet50LegacyYOLO
from utils import util
from utils.config import load_config
from utils.json_dataset import letterbox, list_image_files, scale_boxes_to_original


def parse_args():
    parser = argparse.ArgumentParser(description='Run object detection inference.')
    parser.add_argument('--config', default='config/current_best.yaml')
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', default='./models/best.pth')
    parser.add_argument('--checkpoint2', default=None)
    parser.add_argument('--input_size', default=None, type=int)
    parser.add_argument('--input_size2', default=None, type=int)
    parser.add_argument('--confidence', default=None, type=float)
    parser.add_argument('--confidence2', default=None, type=float)
    parser.add_argument('--iou', default=None, type=float)
    parser.add_argument('--iou2', default=None, type=float)
    parser.add_argument('--max_detections', default=300, type=int)
    parser.add_argument('--max_detections2', default=None, type=int)
    parser.add_argument('--match_iou', default=0.355, type=float)
    parser.add_argument('--unmatched_penalty', default=0.220, type=float)
    parser.add_argument('--ensemble_weight', default=0.50, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint.get('classes', ['person', 'car', 'dog', 'cat', 'chair'])
    model_config = checkpoint.get('config', {}).get('model', {})
    model = ResNet50LegacyYOLO(
        num_classes=len(classes),
        pretrained_backbone=False,
        neck_channels=model_config.get('neck_channels', [64, 128, 256]),
        fpn_depth=model_config.get('fpn_depth', 1),
    )
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    return model, classes, checkpoint.get('input_size', 640)


def detect_image(
    model,
    image,
    classes,
    input_size,
    confidence,
    iou,
    max_detections,
    device,
):
    height, width = image.shape[:2]
    resized, ratio, pad = letterbox(image, input_size)
    tensor = torch.from_numpy(resized.transpose((2, 0, 1))[::-1].copy())
    tensor = tensor.unsqueeze(0).to(device).float() / 255.0

    with torch.no_grad():
        outputs = model(tensor)
        detections = util.non_max_suppression(
            outputs,
            confidence_threshold=confidence,
            iou_threshold=iou,
            max_detections=max_detections,
        )[0]

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

    boxes.sort(key=lambda item: item['confidence'], reverse=True)
    return boxes


def bbox_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    area_a = max(box_a[2] - box_a[0], 0.0) * max(box_a[3] - box_a[1], 0.0)
    area_b = max(box_b[2] - box_b[0], 0.0) * max(box_b[3] - box_b[1], 0.0)
    return intersection / max(area_a + area_b - intersection, 1e-9)


def fuse_detections(
    boxes_a,
    boxes_b,
    match_iou,
    unmatched_penalty,
    weight_a,
    max_detections,
):
    used_b = set()
    fused = []
    weight_b = 1.0 - weight_a

    for box_a in boxes_a:
        best_index = -1
        best_iou = 0.0
        for index, box_b in enumerate(boxes_b):
            if index in used_b or box_a['class'] != box_b['class']:
                continue
            overlap = bbox_iou(box_a['bbox'], box_b['bbox'])
            if overlap > best_iou:
                best_iou = overlap
                best_index = index

        if best_index >= 0 and best_iou >= match_iou:
            used_b.add(best_index)
            box_b = boxes_b[best_index]
            fused.append({
                'class': box_a['class'],
                'confidence': round(math.sqrt(
                    float(box_a['confidence']) * float(box_b['confidence'])
                ), 6),
                'bbox': [
                    round(
                        weight_a * float(value_a) + weight_b * float(value_b),
                        3,
                    )
                    for value_a, value_b in zip(box_a['bbox'], box_b['bbox'])
                ],
            })
        else:
            fused.append({
                **box_a,
                'confidence': round(
                    float(box_a['confidence']) * unmatched_penalty,
                    6,
                ),
            })

    for index, box_b in enumerate(boxes_b):
        if index not in used_b:
            fused.append({
                **box_b,
                'confidence': round(
                    float(box_b['confidence']) * unmatched_penalty,
                    6,
                ),
            })

    fused.sort(key=lambda item: item['confidence'], reverse=True)
    return fused[:max_detections]


def predict_image(
    model,
    image_path,
    classes,
    input_size,
    confidence,
    iou,
    max_detections,
    device,
    ensemble=None,
):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f'Unable to read image: {image_path}')

    boxes = detect_image(
        model,
        image,
        classes,
        input_size,
        confidence,
        iou,
        max_detections,
        device,
    )
    if ensemble is not None:
        boxes_b = detect_image(
            ensemble['model'],
            image,
            classes,
            ensemble['input_size'],
            ensemble['confidence'],
            ensemble['iou'],
            ensemble['max_detections'],
            device,
        )
        boxes = fuse_detections(
            boxes,
            boxes_b,
            ensemble['match_iou'],
            ensemble['unmatched_penalty'],
            ensemble['weight_a'],
            max_detections,
        )

    return {'image_id': image_path.name, 'boxes': boxes}


def run_predictions(
    model,
    image_dir,
    classes,
    input_size,
    confidence,
    iou,
    max_detections,
    device,
    ensemble=None,
):
    predictions = []
    for image_path in list_image_files(image_dir):
        predictions.append(predict_image(
            model,
            image_path,
            classes,
            input_size,
            confidence,
            iou,
            max_detections,
            device,
            ensemble=ensemble,
        ))
    return predictions


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    model, classes, checkpoint_input_size = load_model(args.checkpoint, device)
    input_size = args.input_size or checkpoint_input_size or config['model']['input_size']
    confidence = args.confidence if args.confidence is not None else config['inference']['confidence']
    iou = args.iou if args.iou is not None else config['inference']['iou']
    if args.max_detections < 1:
        raise ValueError('--max_detections must be at least 1')

    ensemble = None
    if args.checkpoint2:
        model2, classes2, checkpoint_input_size2 = load_model(args.checkpoint2, device)
        if classes2 != classes:
            raise ValueError('Both checkpoints must use the same classes')
        max_detections2 = args.max_detections2 or args.max_detections
        if max_detections2 < 1:
            raise ValueError('--max_detections2 must be at least 1')
        if not 0.0 <= args.match_iou <= 1.0:
            raise ValueError('--match_iou must be between 0 and 1')
        if not 0.0 <= args.unmatched_penalty <= 1.0:
            raise ValueError('--unmatched_penalty must be between 0 and 1')
        if not 0.0 <= args.ensemble_weight <= 1.0:
            raise ValueError('--ensemble_weight must be between 0 and 1')
        ensemble = {
            'model': model2,
            'input_size': (
                args.input_size2
                or checkpoint_input_size2
                or input_size
            ),
            'confidence': (
                args.confidence2
                if args.confidence2 is not None
                else confidence
            ),
            'iou': args.iou2 if args.iou2 is not None else iou,
            'max_detections': max_detections2,
            'match_iou': args.match_iou,
            'unmatched_penalty': args.unmatched_penalty,
            'weight_a': args.ensemble_weight,
        }

    predictions = run_predictions(
        model=model,
        image_dir=args.image_dir,
        classes=classes,
        input_size=input_size,
        confidence=confidence,
        iou=iou,
        max_detections=args.max_detections,
        device=device,
        ensemble=ensemble,
    )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'saved predictions to {output_path}')


if __name__ == '__main__':
    main()
