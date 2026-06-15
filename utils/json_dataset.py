import json
import random
from pathlib import Path

import cv2
import numpy
import torch
from torch.utils import data


IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}


def load_annotations(annotation_path):
    with Path(annotation_path).open('r', encoding='utf-8') as file:
        annotation = json.load(file)

    classes = annotation['classes']
    class_to_idx = {name: index for index, name in enumerate(classes)}
    boxes_by_image = {image['id']: [] for image in annotation['images']}

    for item in annotation['annotations']:
        image_id = item['image_id']
        if image_id not in boxes_by_image:
            continue
        boxes_by_image[image_id].append({
            'class_id': class_to_idx[item['class']],
            'bbox': [float(value) for value in item['bbox']],
        })

    return annotation, classes, boxes_by_image


def resolve_image_path(image_dir, file_name):
    image_dir = Path(image_dir)
    file_path = Path(file_name)
    candidate = image_dir / file_path.name
    if candidate.exists():
        return candidate
    return image_dir / file_path


def list_image_files(image_dir):
    image_dir = Path(image_dir)
    files = [path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=lambda path: path.name)


def letterbox(image, input_size, color=(0, 0, 0)):
    height, width = image.shape[:2]
    ratio = min(input_size / height, input_size / width)
    new_width = int(round(width * ratio))
    new_height = int(round(height * ratio))

    pad_w = (input_size - new_width) / 2
    pad_h = (input_size - new_height) / 2

    if (width, height) != (new_width, new_height):
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, ratio, (left, top)


def augment_hsv(image, h_gain=0.015, s_gain=0.7, v_gain=0.4):
    gains = numpy.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    x = numpy.arange(0, 256, dtype=gains.dtype)
    lut_hue = ((x * gains[0]) % 180).astype(numpy.uint8)
    lut_sat = numpy.clip(x * gains[1], 0, 255).astype(numpy.uint8)
    lut_val = numpy.clip(x * gains[2], 0, 255).astype(numpy.uint8)

    hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def random_affine(image, boxes, translate=0.10, scale=0.20):
    size = image.shape[0]
    scale_factor = random.uniform(1 - scale, 1 + scale)
    translate_x = random.uniform(-translate, translate) * size
    translate_y = random.uniform(-translate, translate) * size

    matrix = numpy.array([
        [scale_factor, 0, (1 - scale_factor) * size / 2 + translate_x],
        [0, scale_factor, (1 - scale_factor) * size / 2 + translate_y],
    ], dtype=numpy.float32)

    image = cv2.warpAffine(image, matrix, dsize=(size, size), borderValue=(0, 0, 0))
    if len(boxes):
        corners = numpy.ones((len(boxes) * 4, 3), dtype=numpy.float32)
        corners[:, :2] = boxes[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(len(boxes) * 4, 2)
        transformed = corners @ matrix.T
        transformed = transformed.reshape(len(boxes), 4, 2)
        boxes[:, 0] = transformed[:, :, 0].min(1)
        boxes[:, 1] = transformed[:, :, 1].min(1)
        boxes[:, 2] = transformed[:, :, 0].max(1)
        boxes[:, 3] = transformed[:, :, 1].max(1)

    return image, boxes


def scale_boxes_to_original(boxes, ratio, pad, width, height):
    if boxes.numel() == 0:
        return boxes
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
    return boxes


def build_mosaic(samples, input_size):
    if len(samples) != 4:
        raise ValueError('Mosaic requires exactly four samples')

    tile_size = input_size // 2
    if tile_size < 1 or input_size % 2:
        raise ValueError('Mosaic input size must be a positive even number')

    mosaic = numpy.zeros((input_size, input_size, 3), dtype=numpy.uint8)
    mosaic_boxes = []
    mosaic_labels = []
    offsets = (
        (0, 0),
        (tile_size, 0),
        (0, tile_size),
        (tile_size, tile_size),
    )

    for (image, boxes, labels), (offset_x, offset_y) in zip(samples, offsets):
        tile, ratio, pad = letterbox(image, tile_size)
        mosaic[
            offset_y:offset_y + tile_size,
            offset_x:offset_x + tile_size,
        ] = tile

        if len(boxes):
            transformed = boxes.copy()
            transformed[:, [0, 2]] = (
                transformed[:, [0, 2]] * ratio + pad[0] + offset_x
            )
            transformed[:, [1, 3]] = (
                transformed[:, [1, 3]] * ratio + pad[1] + offset_y
            )
            mosaic_boxes.append(transformed)
            mosaic_labels.append(labels.copy())

    boxes = (
        numpy.concatenate(mosaic_boxes, axis=0)
        if mosaic_boxes
        else numpy.zeros((0, 4), dtype=numpy.float32)
    )
    labels = (
        numpy.concatenate(mosaic_labels, axis=0)
        if mosaic_labels
        else numpy.zeros((0, 1), dtype=numpy.float32)
    )
    return mosaic, boxes, labels


class JsonDetectionDataset(data.Dataset):
    def __init__(self, annotation_path, image_dir, input_size=640, augment=False, params=None):
        self.annotation, self.classes, self.boxes_by_image = load_annotations(annotation_path)
        self.image_dir = Path(image_dir)
        self.input_size = input_size
        self.augment = augment
        self.params = params or {}
        self.images = self.annotation['images']
        self.current_epoch = 0
        self.total_epochs = 1

    def __len__(self):
        return len(self.images)

    def set_epoch(self, epoch, total_epochs):
        self.current_epoch = int(epoch)
        self.total_epochs = max(int(total_epochs), 1)

    def mosaic_probability(self):
        probability = float(self.params.get('mosaic', 0.0))
        close_epochs = max(int(self.params.get('close_mosaic_epochs', 0)), 0)
        if close_epochs and self.current_epoch >= self.total_epochs - close_epochs:
            return 0.0
        return min(max(probability, 0.0), 1.0)

    def load_raw(self, index):
        image_info = self.images[index]
        image_path = resolve_image_path(self.image_dir, image_info['file_name'])
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f'Unable to read image: {image_path}')

        original_h, original_w = image.shape[:2]
        boxes = []
        labels = []
        for item in self.boxes_by_image[image_info['id']]:
            boxes.append(item['bbox'])
            labels.append(item['class_id'])

        boxes = numpy.array(boxes, dtype=numpy.float32).reshape(-1, 4)
        labels = numpy.array(labels, dtype=numpy.float32).reshape(-1, 1)
        return image, boxes, labels

    def __getitem__(self, index):
        use_mosaic = (
            self.augment
            and self.mosaic_probability() > 0.0
            and random.random() < self.mosaic_probability()
        )
        if use_mosaic:
            indices = [index] + [
                random.randrange(len(self.images))
                for _ in range(3)
            ]
            image, boxes, labels = build_mosaic(
                [self.load_raw(sample_index) for sample_index in indices],
                self.input_size,
            )
        else:
            image, boxes, labels = self.load_raw(index)
            image, ratio, pad = letterbox(image, self.input_size)
            if len(boxes):
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio + pad[0]
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio + pad[1]

        if self.augment:
            image, boxes = random_affine(
                image,
                boxes,
                translate=(
                    self.params.get('mosaic_translate', 0.05)
                    if use_mosaic
                    else self.params.get('translate', 0.10)
                ),
                scale=min(
                    self.params.get('mosaic_scale', 0.10)
                    if use_mosaic
                    else self.params.get('scale', 0.20),
                    0.20,
                ),
            )

            if random.random() < self.params.get('flip_lr', 0.5):
                image = numpy.fliplr(image)
                if len(boxes):
                    x1 = boxes[:, 0].copy()
                    x2 = boxes[:, 2].copy()
                    boxes[:, 0] = self.input_size - x2
                    boxes[:, 2] = self.input_size - x1

            image = augment_hsv(
                image,
                self.params.get('hsv_h', 0.015),
                self.params.get('hsv_s', 0.7),
                self.params.get('hsv_v', 0.4),
            )

        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, self.input_size - 1e-3)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, self.input_size - 1e-3)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid]
            labels = labels[valid]

        target_boxes = numpy.zeros((len(boxes), 4), dtype=numpy.float32)
        if len(boxes):
            target_boxes[:, 0] = ((boxes[:, 0] + boxes[:, 2]) / 2) / self.input_size
            target_boxes[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2) / self.input_size
            target_boxes[:, 2] = (boxes[:, 2] - boxes[:, 0]) / self.input_size
            target_boxes[:, 3] = (boxes[:, 3] - boxes[:, 1]) / self.input_size

        sample = image.transpose((2, 0, 1))[::-1]
        sample = numpy.ascontiguousarray(sample)

        return (
            torch.from_numpy(sample),
            torch.from_numpy(labels.astype(numpy.float32)),
            torch.from_numpy(target_boxes),
            torch.zeros(len(target_boxes)),
        )

    @staticmethod
    def collate_fn(batch):
        samples, cls, box, indices = zip(*batch)
        cls = torch.cat(cls, dim=0)
        box = torch.cat(box, dim=0)

        batch_indices = []
        for index, item_indices in enumerate(indices):
            batch_indices.append(item_indices + index)

        targets = {
            'cls': cls,
            'box': box,
            'idx': torch.cat(batch_indices, dim=0),
        }
        return torch.stack(samples, dim=0), targets
