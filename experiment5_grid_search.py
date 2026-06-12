import argparse
import csv
import json
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from predict import load_model
from public.tools.evaluate_predictions import evaluate, load_json, validate_ground_truth
from utils import util
from utils.json_dataset import letterbox, list_image_files, scale_boxes_to_original


DEFAULT_CONFIDENCES = (0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25)
DEFAULT_NMS_IOUS = (0.45, 0.50, 0.55, 0.60, 0.65)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Grid-search confidence and NMS IoU on a fixed detector checkpoint.'
    )
    parser.add_argument('--checkpoint', default='weights/1006-exp1/best.pth')
    parser.add_argument('--image_dir', default='public/val/images')
    parser.add_argument('--ground_truth', default='public/annotations/val.json')
    parser.add_argument('--output_dir', default='weights/1006-exp1/grid_search')
    parser.add_argument('--input_size', default=None, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument(
        '--confidences',
        nargs='+',
        type=float,
        default=list(DEFAULT_CONFIDENCES),
    )
    parser.add_argument(
        '--nms_ious',
        nargs='+',
        type=float,
        default=list(DEFAULT_NMS_IOUS),
    )
    parser.add_argument('--max_detections', default=100, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no_amp', action='store_true')
    return parser.parse_args()


def validate_args(args):
    if args.batch_size < 1:
        raise ValueError('--batch_size must be at least 1.')
    if args.max_detections < 1:
        raise ValueError('--max_detections must be at least 1.')
    if not args.confidences:
        raise ValueError('At least one confidence threshold is required.')
    if not args.nms_ious:
        raise ValueError('At least one NMS IoU threshold is required.')
    if any(value < 0 or value > 1 for value in args.confidences):
        raise ValueError('Confidence thresholds must be between 0 and 1.')
    if any(value < 0 or value > 1 for value in args.nms_ious):
        raise ValueError('NMS IoU thresholds must be between 0 and 1.')

    args.confidences = sorted(set(args.confidences))
    args.nms_ious = sorted(set(args.nms_ious))
    return args


def load_batch(image_paths, input_size, device):
    tensors = []
    metadata = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f'Unable to read image: {image_path}')

        height, width = image.shape[:2]
        resized, ratio, pad = letterbox(image, input_size)
        tensor = torch.from_numpy(resized.transpose((2, 0, 1))[::-1].copy())
        tensors.append(tensor)
        metadata.append((image_path.name, ratio, pad, width, height))

    samples = torch.stack(tensors).to(device, non_blocking=True).float() / 255.0
    return samples, metadata


def compact_detections(detections, metadata, classes):
    image_id, ratio, pad, width, height = metadata
    if not detections.shape[0]:
        return image_id, []

    detections = detections.detach().cpu()
    detections[:, :4] = scale_boxes_to_original(
        detections[:, :4],
        ratio,
        pad,
        width,
        height,
    )

    boxes = []
    for x1, y1, x2, y2, confidence, class_id in detections.tolist():
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((
            round(float(confidence), 6),
            int(class_id),
            round(float(x1), 3),
            round(float(y1), 3),
            round(float(x2), 3),
            round(float(y2), 3),
        ))

    boxes.sort(key=lambda item: item[0], reverse=True)
    return image_id, boxes


def collect_nms_results(model, image_paths, input_size, classes, args, device):
    minimum_confidence = min(args.confidences)
    records = {iou: [] for iou in args.nms_ious}
    amp_enabled = device.type == 'cuda' and not args.no_amp

    progress = tqdm(
        range(0, len(image_paths), args.batch_size),
        desc='Inference and NMS',
        unit='batch',
    )
    with torch.inference_mode():
        for start in progress:
            batch_paths = image_paths[start:start + args.batch_size]
            samples, metadata = load_batch(batch_paths, input_size, device)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(samples)
            outputs = outputs.float()

            for sample_index, sample_metadata in enumerate(metadata):
                sample_output = outputs[sample_index:sample_index + 1]
                for nms_iou in args.nms_ious:
                    detections = util.non_max_suppression(
                        sample_output,
                        confidence_threshold=minimum_confidence,
                        iou_threshold=nms_iou,
                        max_detections=args.max_detections,
                    )[0]
                    records[nms_iou].append(
                        compact_detections(detections, sample_metadata, classes)
                    )

    return records


def build_predictions(records, confidence, classes, max_detections):
    json_predictions = []
    flat_predictions = []

    for image_id, compact_boxes in records:
        selected = [
            box for box in compact_boxes
            if box[0] > confidence
        ][:max_detections]

        image_boxes = []
        for score, class_id, x1, y1, x2, y2 in selected:
            box = {
                'class': classes[class_id],
                'confidence': score,
                'bbox': [x1, y1, x2, y2],
            }
            image_boxes.append(box)
            flat_predictions.append({
                'image_id': image_id,
                **box,
            })

        json_predictions.append({
            'image_id': image_id,
            'boxes': image_boxes,
        })

    return json_predictions, flat_predictions


def result_row(confidence, nms_iou, score, classes):
    row = {
        'confidence': confidence,
        'nms_iou': nms_iou,
        'map50': score['mAP@0.5'],
        'micro_precision': score['micro_precision'],
        'micro_recall': score['micro_recall'],
        'num_predictions': score['num_predictions'],
    }
    for class_name in classes:
        class_score = score['per_class'][class_name]
        row[f'{class_name}_ap'] = class_score['ap']
        row[f'{class_name}_precision'] = class_score['precision']
        row[f'{class_name}_recall'] = class_score['recall']
    return row


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = validate_args(parse_args())
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = load_json(Path(args.ground_truth))
    gt_classes, image_info = validate_ground_truth(ground_truth)

    model, checkpoint_classes, checkpoint_input_size = load_model(args.checkpoint, device)
    if checkpoint_classes != gt_classes:
        raise ValueError(
            f'Checkpoint classes {checkpoint_classes} do not match '
            f'ground-truth classes {gt_classes}.'
        )

    input_size = args.input_size or checkpoint_input_size
    image_paths = list_image_files(args.image_dir)
    expected_images = set(image_info)
    actual_images = {path.name for path in image_paths}
    if actual_images != expected_images:
        missing = sorted(expected_images - actual_images)
        extra = sorted(actual_images - expected_images)
        raise ValueError(
            f'Image directory does not match ground truth. '
            f'Missing={missing[:5]}, extra={extra[:5]}.'
        )

    print(
        f'checkpoint={args.checkpoint} device={device} input_size={input_size} '
        f'images={len(image_paths)} combinations={len(args.confidences) * len(args.nms_ious)}'
    )
    nms_records = collect_nms_results(
        model=model,
        image_paths=image_paths,
        input_size=input_size,
        classes=checkpoint_classes,
        args=args,
        device=device,
    )

    results = []
    best = None
    for nms_iou in args.nms_ious:
        for confidence in args.confidences:
            json_predictions, flat_predictions = build_predictions(
                nms_records[nms_iou],
                confidence,
                checkpoint_classes,
                args.max_detections,
            )
            score = evaluate(
                ground_truth=ground_truth,
                predictions=flat_predictions,
                classes=gt_classes,
                iou_threshold=0.5,
            )
            row = result_row(confidence, nms_iou, score, gt_classes)
            results.append(row)
            print(
                f"conf={confidence:.2f} nms={nms_iou:.2f} "
                f"mAP50={score['mAP@0.5']:.6f} "
                f"P={score['micro_precision']:.6f} "
                f"R={score['micro_recall']:.6f}"
            )

            if best is None or score['mAP@0.5'] > best['score']['mAP@0.5']:
                best = {
                    'confidence': confidence,
                    'nms_iou': nms_iou,
                    'score': score,
                    'predictions': json_predictions,
                }

    results.sort(
        key=lambda item: (
            item['map50'],
            item['micro_recall'],
            item['micro_precision'],
        ),
        reverse=True,
    )
    ranked_results = [
        {'rank': rank, **row}
        for rank, row in enumerate(results, start=1)
    ]

    write_csv(output_dir / 'grid_results.csv', ranked_results)
    summary = {
        'checkpoint': str(Path(args.checkpoint)),
        'image_dir': str(Path(args.image_dir)),
        'ground_truth': str(Path(args.ground_truth)),
        'input_size': input_size,
        'max_detections_per_image': args.max_detections,
        'confidences': args.confidences,
        'nms_ious': args.nms_ious,
        'best': {
            'confidence': best['confidence'],
            'nms_iou': best['nms_iou'],
            'score': best['score'],
        },
        'results': ranked_results,
    }
    (output_dir / 'grid_results.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    (output_dir / 'best_predictions.json').write_text(
        json.dumps(best['predictions'], ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    (output_dir / 'best_score.json').write_text(
        json.dumps(best['score'], ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )

    print(
        f"best confidence={best['confidence']:.2f} "
        f"NMS_IoU={best['nms_iou']:.2f} "
        f"mAP50={best['score']['mAP@0.5']:.6f}"
    )
    print(f'results saved to {output_dir}')


if __name__ == '__main__':
    main()
