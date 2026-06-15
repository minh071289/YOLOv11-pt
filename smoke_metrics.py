import numpy
import torch

from experiment8_horizontal_tta import (
    merge_tta_outputs,
    unflip_horizontal_outputs,
)
from experiment8_multiscale_tta import outputs_to_original
from utils import util


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


def check_horizontal_flip_tta():
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
    assert torch.equal(restored, outputs)

    merged = merge_tta_outputs(outputs, outputs, 32)
    assert merged.shape == (1, 5, 4)


def check_multiscale_tta():
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
    assert torch.allclose(converted, expected)


if __name__ == '__main__':
    check_compute_ap()
    check_compute_metric()
    check_compute_ciou()
    check_non_max_suppression()
    check_horizontal_flip_tta()
    check_multiscale_tta()
    print('metric smoke checks passed')
