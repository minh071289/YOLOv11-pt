import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fuse two prediction files using cross-model consensus.'
    )
    parser.add_argument('--predictions_a', required=True, type=Path)
    parser.add_argument('--predictions_b', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--match_iou', default=0.355, type=float)
    parser.add_argument('--unmatched_penalty', default=0.220, type=float)
    parser.add_argument('--weight_a', default=0.50, type=float)
    parser.add_argument('--max_detections', default=300, type=int)
    return parser.parse_args()


def bbox_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    area_a = max(box_a[2] - box_a[0], 0.0) * max(box_a[3] - box_a[1], 0.0)
    area_b = max(box_b[2] - box_b[0], 0.0) * max(box_b[3] - box_b[1], 0.0)
    return intersection / max(area_a + area_b - intersection, 1e-9)


def load_predictions(path):
    with path.open('r', encoding='utf-8') as file:
        predictions = json.load(file)
    return {entry['image_id']: entry['boxes'] for entry in predictions}


def fuse_image(boxes_a, boxes_b, match_iou, unmatched_penalty, weight_a):
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
                'confidence': math.sqrt(
                    float(box_a['confidence']) * float(box_b['confidence'])
                ),
                'bbox': [
                    weight_a * float(value_a) + weight_b * float(value_b)
                    for value_a, value_b in zip(box_a['bbox'], box_b['bbox'])
                ],
            })
        else:
            fused.append({
                **box_a,
                'confidence': float(box_a['confidence']) * unmatched_penalty,
            })

    for index, box_b in enumerate(boxes_b):
        if index not in used_b:
            fused.append({
                **box_b,
                'confidence': float(box_b['confidence']) * unmatched_penalty,
            })

    return sorted(fused, key=lambda item: item['confidence'], reverse=True)


def main():
    args = parse_args()
    predictions_a = load_predictions(args.predictions_a)
    predictions_b = load_predictions(args.predictions_b)

    if predictions_a.keys() != predictions_b.keys():
        raise ValueError('Prediction files must contain the same image IDs')
    if not 0.0 <= args.weight_a <= 1.0:
        raise ValueError('--weight_a must be between 0 and 1')

    output = []
    for image_id, boxes_a in predictions_a.items():
        boxes = fuse_image(
            boxes_a,
            predictions_b[image_id],
            match_iou=args.match_iou,
            unmatched_penalty=args.unmatched_penalty,
            weight_a=args.weight_a,
        )
        output.append({
            'image_id': image_id,
            'boxes': boxes[:args.max_detections],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as file:
        json.dump(output, file, ensure_ascii=False)


if __name__ == '__main__':
    main()
