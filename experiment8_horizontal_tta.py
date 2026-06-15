import argparse
import json
import time
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from predict import load_model
from public.tools.evaluate_predictions import (
    evaluate,
    load_json,
    normalize_predictions,
    validate_ground_truth,
)
from utils import util
from utils.json_dataset import (
    letterbox,
    list_image_files,
    scale_boxes_to_original,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate horizontal flip-TTA on a fixed detector checkpoint.'
    )
    parser.add_argument('--checkpoint', default='weights/1306-exp8/best.pth')
    parser.add_argument('--image_dir', default='public/val/images')
    parser.add_argument('--ground_truth', default='public/annotations/val.json')
    parser.add_argument('--output_dir', default='weights/1306-exp8/tta_horizontal')
    parser.add_argument('--baseline_score', default='weights/1306-exp8/grid_search_extended_max300/best_score.json')
    parser.add_argument('--input_size', default=None, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--confidence', default=0.0001, type=float)
    parser.add_argument('--nms_iou', default=0.50, type=float)
    parser.add_argument('--max_detections', default=300, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no_amp', action='store_true')
    return parser.parse_args()


def validate_args(args):
    if args.batch_size < 1:
        raise ValueError('--batch_size must be at least 1')
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError('--confidence must be between 0 and 1')
    if not 0.0 <= args.nms_iou <= 1.0:
        raise ValueError('--nms_iou must be between 0 and 1')
    if args.max_detections < 1:
        raise ValueError('--max_detections must be at least 1')
    return args


def unflip_horizontal_outputs(outputs, input_size):
    unflipped = outputs.clone()
    unflipped[:, 0, :] = float(input_size) - unflipped[:, 0, :]
    return unflipped


def merge_tta_outputs(original_outputs, flipped_outputs, input_size):
    unflipped_outputs = unflip_horizontal_outputs(flipped_outputs, input_size)
    return torch.cat((original_outputs, unflipped_outputs), dim=2)


def run_self_tests():
    outputs = torch.tensor([[
        [10.0, 20.0],
        [5.0, 15.0],
        [4.0, 8.0],
        [6.0, 10.0],
        [0.9, 0.8],
    ]])
    restored = unflip_horizontal_outputs(
        unflip_horizontal_outputs(outputs, 32),
        32,
    )
    if not torch.equal(restored, outputs):
        raise AssertionError('Horizontal box transform is not self-inverse')

    merged = merge_tta_outputs(outputs, outputs, 32)
    if merged.shape[2] != outputs.shape[2] * 2:
        raise AssertionError('TTA outputs must be merged before NMS')


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


def detections_to_boxes(detections, metadata, classes):
    image_id, ratio, pad, width, height = metadata
    if not detections.shape[0]:
        return {'image_id': image_id, 'boxes': []}

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
        boxes.append({
            'class': classes[int(class_id)],
            'confidence': round(float(confidence), 6),
            'bbox': [
                round(float(x1), 3),
                round(float(y1), 3),
                round(float(x2), 3),
                round(float(y2), 3),
            ],
        })

    boxes.sort(key=lambda item: item['confidence'], reverse=True)
    return {'image_id': image_id, 'boxes': boxes}


def run_tta(model, image_paths, input_size, classes, args, device):
    predictions = []
    amp_enabled = device.type == 'cuda' and not args.no_amp
    inference_seconds = 0.0

    progress = tqdm(
        range(0, len(image_paths), args.batch_size),
        desc='Horizontal flip-TTA',
        unit='batch',
    )
    with torch.inference_mode():
        for start in progress:
            batch_paths = image_paths[start:start + args.batch_size]
            samples, metadata = load_batch(batch_paths, input_size, device)

            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            inference_start = time.perf_counter()
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                original_outputs = model(samples)
                flipped_outputs = model(torch.flip(samples, dims=(3,)))
            merged_outputs = merge_tta_outputs(
                original_outputs.float(),
                flipped_outputs.float(),
                input_size,
            )
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - inference_start

            for sample_index, sample_metadata in enumerate(metadata):
                detections = util.non_max_suppression(
                    merged_outputs[sample_index:sample_index + 1],
                    confidence_threshold=args.confidence,
                    iou_threshold=args.nms_iou,
                    max_detections=args.max_detections,
                )[0]
                if detections.shape[0] > args.max_detections:
                    raise AssertionError('NMS exceeded max_detections')
                predictions.append(
                    detections_to_boxes(detections, sample_metadata, classes)
                )

    return predictions, inference_seconds


def metric_delta(score, baseline):
    return {
        'mAP@0.5': round(score['mAP@0.5'] - baseline['mAP@0.5'], 6),
        'micro_precision': round(
            score['micro_precision'] - baseline['micro_precision'],
            6,
        ),
        'micro_recall': round(
            score['micro_recall'] - baseline['micro_recall'],
            6,
        ),
        'num_predictions': score['num_predictions'] - baseline['num_predictions'],
        'per_class_ap': {
            class_name: round(
                score['per_class'][class_name]['ap']
                - baseline['per_class'][class_name]['ap'],
                6,
            )
            for class_name in score['per_class']
        },
    }


def main():
    args = validate_args(parse_args())
    run_self_tests()

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = load_json(Path(args.ground_truth))
    classes, image_info = validate_ground_truth(ground_truth)
    image_paths = list_image_files(args.image_dir)
    expected_images = set(image_info)
    actual_images = {path.name for path in image_paths}
    if actual_images != expected_images:
        missing = sorted(expected_images - actual_images)
        extra = sorted(actual_images - expected_images)
        raise ValueError(
            f'Image directory does not match ground truth. '
            f'Missing={missing[:5]}, extra={extra[:5]}'
        )

    model, checkpoint_classes, checkpoint_input_size = load_model(
        args.checkpoint,
        device,
    )
    if checkpoint_classes != classes:
        raise ValueError(
            f'Checkpoint classes {checkpoint_classes} do not match '
            f'ground-truth classes {classes}'
        )
    input_size = args.input_size or checkpoint_input_size

    wall_start = time.perf_counter()
    predictions, inference_seconds = run_tta(
        model,
        image_paths,
        input_size,
        classes,
        args,
        device,
    )
    wall_seconds = time.perf_counter() - wall_start

    predictions_path = output_dir / 'tta_predictions.json'
    predictions_path.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )

    normalized_predictions = normalize_predictions(
        predictions,
        classes=classes,
        image_info=image_info,
        max_detections_per_image=args.max_detections,
        require_complete=True,
    )
    score = evaluate(
        ground_truth=ground_truth,
        predictions=normalized_predictions,
        classes=classes,
        iou_threshold=0.5,
    )
    score_path = output_dir / 'tta_score.json'
    score_path.write_text(
        json.dumps(score, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )

    baseline = load_json(Path(args.baseline_score))
    counts = [len(item['boxes']) for item in predictions]
    summary = {
        'checkpoint': str(Path(args.checkpoint)),
        'image_dir': str(Path(args.image_dir)),
        'ground_truth': str(Path(args.ground_truth)),
        'device': str(device),
        'amp_enabled': device.type == 'cuda' and not args.no_amp,
        'input_size': input_size,
        'tta': 'horizontal_flip',
        'confidence': args.confidence,
        'nms_iou': args.nms_iou,
        'max_detections_per_image': args.max_detections,
        'num_images': len(image_paths),
        'max_predictions_in_one_image': max(counts, default=0),
        'wall_seconds': round(wall_seconds, 3),
        'wall_ms_per_image': round(wall_seconds * 1000 / len(image_paths), 3),
        'model_inference_seconds': round(inference_seconds, 3),
        'model_inference_ms_per_image': round(
            inference_seconds * 1000 / len(image_paths),
            3,
        ),
        'baseline_score_path': str(Path(args.baseline_score)),
        'baseline': baseline,
        'score': score,
        'delta': metric_delta(score, baseline),
        'target_reached': score['mAP@0.5'] >= 0.85,
    }
    (output_dir / 'tta_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )

    print(json.dumps(score, ensure_ascii=False, indent=2))
    print(f"baseline mAP@0.5={baseline['mAP@0.5']:.6f}")
    print(f"TTA delta={summary['delta']['mAP@0.5']:+.6f}")
    print(f"target_reached={summary['target_reached']}")
    print(f'artifacts saved to {output_dir}')


if __name__ == '__main__':
    main()
