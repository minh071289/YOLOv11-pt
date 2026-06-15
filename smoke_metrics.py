import numpy
import torch

from utils import util
from utils.json_dataset import JsonDetectionDataset, build_mosaic


def check_compute_metric():
    iou_v = torch.linspace(0.5, 0.95, 10)
    output = torch.tensor([
        [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
        [20.0, 20.0, 30.0, 30.0, 0.8, 1.0],
    ])
    target = torch.tensor([[0.0, 0.0, 0.0, 10.0, 10.0]])
    metric = util.compute_metric(output, target, iou_v)
    assert metric.shape == (2, 10)
    assert metric[0].all().item()
    assert not metric[1].any().item()


def check_compute_ap():
    tp = numpy.array([
        [True] * 10,
        [False] * 10,
    ])
    conf = numpy.array([0.90, 0.20])
    pred_cls = numpy.array([0.0, 1.0])
    target_cls = numpy.array([0.0])
    _, _, precision, recall, map50, mean_ap = util.compute_ap(tp, conf, pred_cls, target_cls)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= map50 <= 1.0
    assert 0.0 <= mean_ap <= 1.0
    assert map50 > 0.95


def check_compute_ciou():
    same = util.compute_ciou(
        torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
    )
    shifted = util.compute_ciou(
        torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        torch.tensor([[5.0, 5.0, 15.0, 15.0]]),
    )
    disjoint = util.compute_ciou(
        torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        torch.tensor([[20.0, 20.0, 30.0, 30.0]]),
    )
    assert torch.isclose(same, torch.ones_like(same), atol=1e-5).all().item()
    assert (shifted < same).all().item()
    assert (disjoint < 0).all().item()


def check_non_max_suppression():
    # Shape: batch x (4 box channels + class channels) x anchors.
    # Boxes are cx, cy, w, h. The first two overlap heavily in class 0.
    outputs = torch.tensor([[
        [10.0, 10.5, 50.0, 10.0],
        [10.0, 10.5, 50.0, 10.0],
        [10.0, 10.0, 12.0, 10.0],
        [10.0, 10.0, 12.0, 10.0],
        [0.90, 0.80, 0.95, 0.05],
        [0.05, 0.05, 0.05, 0.90],
    ]])
    detections = util.non_max_suppression(outputs, confidence_threshold=0.25, iou_threshold=0.50)[0]
    assert detections.shape[1] == 6
    assert detections.shape[0] == 3
    classes = set(detections[:, 5].tolist())
    assert classes == {0.0, 1.0}
    assert abs(detections[:, 4].max().item() - 0.95) < 1e-6


def check_controlled_mosaic():
    samples = []
    for class_id in range(4):
        image = numpy.full((20, 40, 3), class_id * 50, dtype=numpy.uint8)
        boxes = numpy.array([[0.0, 0.0, 40.0, 20.0]], dtype=numpy.float32)
        labels = numpy.array([[class_id]], dtype=numpy.float32)
        samples.append((image, boxes, labels))

    image, boxes, labels = build_mosaic(samples, input_size=64)
    assert image.shape == (64, 64, 3)
    assert boxes.shape == (4, 4)
    assert labels[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert (boxes >= 0).all()
    assert (boxes <= 64).all()
    assert (boxes[:, 2] > boxes[:, 0]).all()
    assert (boxes[:, 3] > boxes[:, 1]).all()

    dataset = JsonDetectionDataset.__new__(JsonDetectionDataset)
    dataset.params = {'mosaic': 0.25, 'close_mosaic_epochs': 10}
    dataset.set_epoch(39, 50)
    assert dataset.mosaic_probability() == 0.25
    dataset.set_epoch(40, 50)
    assert dataset.mosaic_probability() == 0.0


if __name__ == '__main__':
    check_compute_ap()
    check_compute_metric()
    check_compute_ciou()
    check_non_max_suppression()
    check_controlled_mosaic()
    print('metric smoke checks passed')
