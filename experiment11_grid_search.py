import argparse
import csv
import json
import math
from pathlib import Path

from public.tools.evaluate_predictions import evaluate, load_json, validate_ground_truth

from experiment11_consensus_ensemble import bbox_iou, load_predictions


DEFAULT_MATCH_IOUS = tuple(round(0.20 + 0.025 * index, 3) for index in range(13))
DEFAULT_UNMATCHED_PENALTIES = tuple(
    round(0.05 + 0.025 * index, 3) for index in range(15)
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Ablate consensus match IoU and unmatched-score penalty.'
    )
    parser.add_argument(
        '--predictions_a',
        default='weights/1306-exp8/grid_search_extended_max300/best_predictions.json',
        type=Path,
    )
    parser.add_argument(
        '--predictions_b',
        default='weights/1006-exp6/grid_search_extended/best_predictions.json',
        type=Path,
    )
    parser.add_argument(
        '--ground_truth',
        default='public/annotations/val.json',
        type=Path,
    )
    parser.add_argument(
        '--output_dir',
        default='weights/experiment11_consensus_grid',
        type=Path,
    )
    parser.add_argument(
        '--match_ious',
        nargs='+',
        default=list(DEFAULT_MATCH_IOUS),
        type=float,
    )
    parser.add_argument(
        '--unmatched_penalties',
        nargs='+',
        default=list(DEFAULT_UNMATCHED_PENALTIES),
        type=float,
    )
    parser.add_argument('--weight_a', default=0.50, type=float)
    parser.add_argument('--max_detections', default=300, type=int)
    return parser.parse_args()


def validate_args(args):
    if not args.match_ious or not args.unmatched_penalties:
        raise ValueError('Both parameter grids must contain at least one value')
    if any(value < 0.0 or value > 1.0 for value in args.match_ious):
        raise ValueError('--match_ious values must be between 0 and 1')
    if any(value < 0.0 or value > 1.0 for value in args.unmatched_penalties):
        raise ValueError('--unmatched_penalties values must be between 0 and 1')
    if not 0.0 <= args.weight_a <= 1.0:
        raise ValueError('--weight_a must be between 0 and 1')
    if args.max_detections < 1:
        raise ValueError('--max_detections must be at least 1')
    args.match_ious = sorted(set(args.match_ious))
    args.unmatched_penalties = sorted(set(args.unmatched_penalties))
    return args


def build_match_cache(predictions_a, predictions_b):
    cache = {}
    for image_id, boxes_a in predictions_a.items():
        boxes_b = predictions_b[image_id]
        candidates = []
        for box_a in boxes_a:
            matches = [
                (bbox_iou(box_a['bbox'], box_b['bbox']), index)
                for index, box_b in enumerate(boxes_b)
                if box_a['class'] == box_b['class']
            ]
            matches.sort(reverse=True)
            candidates.append(matches)
        cache[image_id] = (boxes_a, boxes_b, candidates)
    return cache


def fuse_cached_image(
    boxes_a,
    boxes_b,
    candidates,
    match_iou,
    unmatched_penalty,
    weight_a,
):
    used_b = set()
    fused = []
    weight_b = 1.0 - weight_a

    for box_a, matches in zip(boxes_a, candidates):
        best_index = -1
        for overlap, index in matches:
            if overlap < match_iou:
                break
            if index not in used_b:
                best_index = index
                break

        if best_index >= 0:
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

    fused.sort(key=lambda item: item['confidence'], reverse=True)
    return fused


def build_predictions(cache, match_iou, unmatched_penalty, weight_a, max_detections):
    document = []
    flat = []
    for image_id, (boxes_a, boxes_b, candidates) in cache.items():
        boxes = fuse_cached_image(
            boxes_a,
            boxes_b,
            candidates,
            match_iou,
            unmatched_penalty,
            weight_a,
        )[:max_detections]
        document.append({'image_id': image_id, 'boxes': boxes})
        flat.extend({'image_id': image_id, **box} for box in boxes)
    return document, flat


def result_row(match_iou, unmatched_penalty, score, classes):
    row = {
        'match_iou': match_iou,
        'unmatched_penalty': unmatched_penalty,
        'map50': score['mAP@0.5'],
        'micro_precision': score['micro_precision'],
        'micro_recall': score['micro_recall'],
        'num_predictions': score['num_predictions'],
    }
    for class_name in classes:
        row[f'{class_name}_ap'] = score['per_class'][class_name]['ap']
    return row


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = validate_args(parse_args())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = load_json(args.ground_truth)
    classes, image_info = validate_ground_truth(ground_truth)
    predictions_a = load_predictions(args.predictions_a)
    predictions_b = load_predictions(args.predictions_b)
    expected_ids = set(image_info)
    if set(predictions_a) != expected_ids or set(predictions_b) != expected_ids:
        raise ValueError('Prediction image IDs must exactly match ground truth')

    combinations = len(args.match_ious) * len(args.unmatched_penalties)
    print(
        f'images={len(expected_ids)} combinations={combinations} '
        f'max_detections={args.max_detections}'
    )
    cache = build_match_cache(predictions_a, predictions_b)

    results = []
    best = None
    for match_iou in args.match_ious:
        for unmatched_penalty in args.unmatched_penalties:
            document, flat = build_predictions(
                cache,
                match_iou,
                unmatched_penalty,
                args.weight_a,
                args.max_detections,
            )
            score = evaluate(
                ground_truth=ground_truth,
                predictions=flat,
                classes=classes,
                iou_threshold=0.5,
            )
            row = result_row(
                match_iou,
                unmatched_penalty,
                score,
                classes,
            )
            results.append(row)
            if best is None or row['map50'] > best['row']['map50']:
                best = {'row': row, 'score': score, 'predictions': document}
                print(
                    f"best map50={row['map50']:.6f} "
                    f'match_iou={match_iou:.3f} '
                    f'unmatched_penalty={unmatched_penalty:.3f}'
                )

    results.sort(key=lambda row: row['map50'], reverse=True)
    write_csv(args.output_dir / 'grid_results.csv', results)
    with (args.output_dir / 'grid_results.json').open('w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with (args.output_dir / 'best_score.json').open('w', encoding='utf-8') as file:
        json.dump(best['score'], file, ensure_ascii=False, indent=2)
    with (args.output_dir / 'best_predictions.json').open('w', encoding='utf-8') as file:
        json.dump(best['predictions'], file, ensure_ascii=False)
    with (args.output_dir / 'best_params.json').open('w', encoding='utf-8') as file:
        json.dump(best['row'], file, ensure_ascii=False, indent=2)

    print(json.dumps(best['row'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
