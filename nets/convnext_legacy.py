import torch
import torchvision.models as tv_models

from nets import legacy_nn


def load_legacy_checkpoint(weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    return checkpoint['model'].float()


class ConvNeXtBaseBackbone(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = tv_models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.convnext_base(weights=weights)
        self.features = model.features
        self.out_channels = (256, 512, 1024)

    def forward(self, x):
        x = self.features[1](self.features[0](x))
        p3 = self.features[3](self.features[2](x))
        p4 = self.features[5](self.features[4](p3))
        p5 = self.features[7](self.features[6](p4))
        return p3, p4, p5


class ConvNeXtLegacyYOLO(torch.nn.Module):
    def __init__(self, num_classes=80, pretrained_backbone=True, checkpoint_path=None, device='cpu'):
        super().__init__()
        self.backbone = ConvNeXtBaseBackbone(pretrained=pretrained_backbone)
        # ConvNeXt-Base emits 256/512/1024 at strides 8/16/32. The legacy nano
        # FPN expects backbone features of 128/128/256 at those same strides.
        self.adapters = torch.nn.ModuleList([
            torch.nn.Conv2d(256, 128, kernel_size=1, bias=True),
            torch.nn.Conv2d(512, 128, kernel_size=1, bias=True),
            torch.nn.Conv2d(1024, 256, kernel_size=1, bias=True),
        ])

        width = [3, 16, 32, 64, 128, 256]
        depth = [1, 1, 1, 1, 1, 1]
        csp = [False, True]

        self.fpn = legacy_nn.DarkFPN(width, depth, csp)
        self.head = legacy_nn.Head(num_classes, (width[3], width[4], width[5]))

        self._initialize_head_stride()

        if checkpoint_path:
            legacy_model = load_legacy_checkpoint(checkpoint_path, device)
            self.fpn.load_state_dict(legacy_model.fpn.state_dict(), strict=True)
            self.head.load_state_dict(legacy_model.head.state_dict(), strict=True)
            self.head.stride = legacy_model.head.stride
            self.stride = self.head.stride

    def _initialize_head_stride(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            p3, p4, p5 = self.forward_features(dummy)
            self.head.stride = torch.tensor([256 / x.shape[-2] for x in (p3, p4, p5)])
            self.stride = self.head.stride
            self.head.initialize_biases()

    def forward_features(self, x):
        p3, p4, p5 = self.backbone(x)
        p3 = self.adapters[0](p3)
        p4 = self.adapters[1](p4)
        p5 = self.adapters[2](p5)
        return self.fpn((p3, p4, p5))

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(list(x))
