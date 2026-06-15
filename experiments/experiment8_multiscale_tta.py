import argparse
import json
import time
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from experiment8_horizontal_tta import metric_delta
from predict import load_model
from public.tools.evaluate_predictions import (
    evaluate,
    load_json,
    normalize_predictions,
    validate_ground_truth,
)
from utils import util
from utils.json_dataset import letterbox, list_image_files


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate single-scale 768 and multi-scale 640+768 TTA.'
    )
    parser.add_argument('--checkpoint', default='weights/1306-exp8/best.pth')
    parser.add_argument('--image_dir', default='public/val/images')
    parser.add_argument('--ground_truth', default='public/annotations/val.json')
    parser.add_argument('--output_dir', default='weights/1306-exp8/tta_multiscale')
    parser.add_argument(
        '--baseline_score',
        default='weights/1306-exp8/grid_search_extended_max300/best_score.json',
    )
    parser.add_argument('--base_size', default=640, type=int)
    parser.add_argument('--large_size', default=768, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--confidence', default=0.0001, type=float)
    parser.add_argument('--nms_iou', default=0.50, type=float)
    parser.add_argument('--max_detections', default=300, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no_amp', action='store_true')
    return parser.parse_args()


def validate_args(args):
    if args.base_size < 1 or args.large_size < 1:
        raise ValueError('Input sizes must be positive')
    if args.base_size == args.large_size:
        raise ValueError('Base and large input sizes must differ')
    if args.batch_size < 1:
        raise ValueError('--batch_size must be at least 1')
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError('--confidence must be between 0 and 1')
    if not 0.0 <= args.nms_iou <= 1.0:
        raise ValueError('--nms_iou must be between 0 and 1')
    if args.max_detections < 1:
        raise ValueError('--max_detections must be at least 1')
    return args


def outputs_to_original(outputs, ratio, pad):
    converted = outputs.clone()
    converted[:, 0, :] = (converted[:, 0, :] - pad[0]) / ratio
    converted[:, 1, :] = (converted[:, 1, :] - pad[1]) / ratio
    converted[:, 2:4, :] /= ratio
    return converted


def run_self_tests():
    outputs = torch.tensor([[
        [60.0],
        [45.0],
        [20.0],
        [10.0],
        [0.9],
    ]])
    converted = outputs_to_original(outputs, ratio=2.0, pad=(10.0, 5.0))
    expected = torch.tensor([[
        [25.0],
        [20.0],
        [10.0],
        [5.0],
        [0.9],
    ]])
    if not torch.allclose(converted, expected):
        raise AssertionError('Scale-to-original coordinate transform failed')


def load_multiscale_batch(image_paths, sizes, device):
    tensors = {size: [] for size in sizes}
    metadata = {size: [] for size in sizes}

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f'Unable to read image: {image_path}')
        height, width = image.shape[:2]

        for size in sizes:
            resized, ratio, pad = letterbox(image, size)
            tensor = torch.from_numpy(resized.transpose((2, 0, 1))[::-1].copy())
            tensors[size].append(tensor)
            metadata[size].append(
                (image_path.name, ratio, pad, width, height)
            )

    tensors = {
        size: torch.stack(items).to(device, non_blocking=True).float() / 255.0
        for size, items in tensors.items()
    }
    return tensors, metadata


def detections_to_prediction(detections, image_info, classes):
    image_id, _, _, width, height = image_info
    boxes = []
    for x1, y1, x2, y2, confidence, class_id in detections.detach().cpu().tolist():
        x1 = min(max(float(x1), 0.0), float(width))
        y1 = min(max(float(y1), 0.0), float(height))
        x2 = min(max(float(x2), 0.0), float(width))
        y2 = min(max(float(y2), 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append({
            'class': classes[int(class_id)],
            'confidence': round(float(confidence), 6),
            'bbox': [
                round(x1, 3),
                round(y1, 3),
                round(x2, 3),
                round(y2, 3),
            ],
        })
    boxes.sort(key=lambda item: item['confidence'], reverse=True)
    return {'image_id': image_id, 'boxes': boxes}


def nms_prediction(outputs, image_info, classes, args):
    detections = util.non_max_suppression(
        outputs,
        confidence_threshold=args.confidence,
        iou_threshold=args.nms_iou,
        max_detections=args.max_detections,
    )[0]
    if detections.shape[0] > args.max_detections:
        raise AssertionError('NMS exceeded max_detections')
    return detections_to_prediction(detections, image_info, classes)


def run_inference(model, image_paths, classes, args, device):
    sizes = (args.base_size, args.large_size)
    single_large_predictions = []
    multiscale_predictions = []
    inference_seconds = 0.0
    amp_enabled = device.type == 'cuda' and not args.no_amp

    progress = tqdm(
        range(0, len(image_paths), args.batch_size),
        desc='Multi-scale TTA',
        unit='batch',
    )
    with torch.inference_mode():
        for start in progress:
            batch_paths = image_paths[start:start + args.batch_size]
            samples, metadata = load_multiscale_batch(batch_paths, sizes, device)

            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            inference_start = time.perf_counter()
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                base_outputs = model(samples[args.base_size])
                large_outputs = model(samples[args.large_size])
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - inference_start

            for sample_index in range(len(batch_paths)):
                base_info = metadata[args.base_size][sample_index]
                large_info = metadata[args.large_size][sample_index]
                _, base_ratio, base_pad, _, _ = base_info
                _, large_ratio, large_pad, _, _ = large_info

                base_original = outputs_to_original(
                    base_outputs[sample_index:sample_index + 1].float(),
                    base_ratio,
                    base_pad,
                )
                large_original = outputs_to_original(
                    large_outputs[sample_index:sample_index + 1].float(),
                    large_ratio,
                    large_pad,
                )

                single_large_predictions.append(
                    nms_prediction(
                        large_original,
                        large_info,
                        classes,
                        args,
                    )
                )
                multiscale_predictions.append(
                    nms_prediction(
                        torch.cat((base_original, large_original), dim=2),
                        base_info,
                        classes,
                        args,
                    )
                )

    return single_large_predictions, multiscale_predictions, inference_seconds


def evaluate_predictions(predictions, ground_truth, classes, image_info, args):
    normalized = normalize_predictions(
        predictions,
        classes=classes,
        image_info=image_info,
        max_detections_per_image=args.max_detections,
        require_complete=True,
    )
    return evaluate(
        ground_truth=ground_truth,
        predictions=normalized,
        classes=classes,
        iou_threshold=0.5,
    )


def write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )


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
    if {path.name for path in image_paths} != set(image_info):
        raise ValueError('Image directory does not match ground truth')

    model, checkpoint_classes, _ = load_model(args.checkpoint, device)
    if checkpoint_classes != classes:
        raise ValueError('Checkpoint classes do not match ground truth')

    wall_start = time.perf_counter()
    single_large, multiscale, inference_seconds = run_inference(
        model,
        image_paths,
        classes,
        args,
        device,
    )
    wall_seconds = time.perf_counter() - wall_start

    single_score = evaluate_predictions(
        single_large,
        ground_truth,
        classes,
        image_info,
        args,
    )
    multiscale_score = evaluate_predictions(
        multiscale,
        ground_truth,
        classes,
        image_info,
        args,
    )
    baseline = load_json(Path(args.baseline_score))

    write_json(output_dir / 'single_768_predictions.json', single_large)
    write_json(output_dir / 'single_768_score.json', single_score)
    write_json(output_dir / 'multiscale_predictions.json', multiscale)
    write_json(output_dir / 'multiscale_score.json', multiscale_score)

    summary = {
        'checkpoint': str(Path(args.checkpoint)),
        'device': str(device),
        'amp_enabled': device.type == 'cuda' and not args.no_amp,
        'base_size': args.base_size,
        'large_size': args.large_size,
        'confidence': args.confidence,
        'nms_iou': args.nms_iou,
        'max_detections_per_image': args.max_detections,
        'num_images': len(image_paths),
        'wall_seconds': round(wall_seconds, 3),
        'wall_ms_per_image': round(wall_seconds * 1000 / len(image_paths), 3),
        'model_inference_seconds': round(inference_seconds, 3),
        'model_inference_ms_per_image': round(
            inference_seconds * 1000 / len(image_paths),
            3,
        ),
        'baseline': baseline,
        'single_768': {
            'score': single_score,
            'delta': metric_delta(single_score, baseline),
        },
        'multiscale_640_768': {
            'score': multiscale_score,
            'delta': metric_delta(multiscale_score, baseline),
        },
        'target_reached': max(
            single_score['mAP@0.5'],
            multiscale_score['mAP@0.5'],
        ) >= 0.85,
    }
    write_json(output_dir / 'tta_summary.json', summary)

    print(f"baseline={baseline['mAP@0.5']:.6f}")
    print(
        f"single_768={single_score['mAP@0.5']:.6f} "
        f"delta={summary['single_768']['delta']['mAP@0.5']:+.6f}"
    )
    print(
        f"multiscale_640_768={multiscale_score['mAP@0.5']:.6f} "
        f"delta={summary['multiscale_640_768']['delta']['mAP@0.5']:+.6f}"
    )
    print(f"target_reached={summary['target_reached']}")
    print(f'artifacts saved to {output_dir}')


if __name__ == '__main__':
    main()
