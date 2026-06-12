import torch

from nets import legacy_nn
from .backbones import ResNet50Backbone


class ResNet50LegacyYOLO(torch.nn.Module):
    def __init__(
        self,
        num_classes=5,
        pretrained_backbone=True,
        neck_channels=(64, 128, 256),
        fpn_depth=1,
    ):
        super().__init__()
        if len(neck_channels) != 3:
            raise ValueError('neck_channels must contain P3, P4, and P5 channels')
        neck_channels = tuple(int(channel) for channel in neck_channels)
        if any(channel <= 0 for channel in neck_channels):
            raise ValueError('neck_channels must be positive')
        if int(fpn_depth) < 1:
            raise ValueError('fpn_depth must be at least 1')

        self.backbone = ResNet50Backbone(pretrained=pretrained_backbone)
        self.adapters = torch.nn.ModuleList([
            torch.nn.Conv2d(512, neck_channels[1], kernel_size=1, bias=True),
            torch.nn.Conv2d(1024, neck_channels[1], kernel_size=1, bias=True),
            torch.nn.Conv2d(2048, neck_channels[2], kernel_size=1, bias=True),
        ])

        width = [3, 16, 32, *neck_channels]
        depth = [1, 1, 1, 1, 1, int(fpn_depth)]
        csp = [False, True]

        self.neck_channels = neck_channels
        self.fpn_depth = int(fpn_depth)
        self.fpn = legacy_nn.DarkFPN(width, depth, csp)
        self.head = legacy_nn.Head(num_classes, neck_channels)
        self.register_buffer('image_mean', torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

        self._initialize_head_stride()

    def _initialize_head_stride(self):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            p3, p4, p5 = self.forward_features(dummy)
            self.head.stride = torch.tensor([256 / x.shape[-2] for x in (p3, p4, p5)])
            self.stride = self.head.stride
            self.head.initialize_biases()
        self.train(was_training)

    def normalize(self, x):
        return (x - self.image_mean.to(x.dtype)) / self.image_std.to(x.dtype)

    def forward_features(self, x):
        x = self.normalize(x)
        p3, p4, p5 = self.backbone(x)
        p3 = self.adapters[0](p3)
        p4 = self.adapters[1](p4)
        p5 = self.adapters[2](p5)
        return self.fpn((p3, p4, p5))

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(list(x))

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_layer4(self):
        for parameter in self.backbone.layer4.parameters():
            parameter.requires_grad = True

    def unfreeze_layer3(self):
        for parameter in self.backbone.layer3.parameters():
            parameter.requires_grad = True

    def set_backbone_train_mode(self):
        # Trainable backbone convolutions still receive gradients in eval mode,
        # while pretrained BatchNorm statistics stay fixed for small batches.
        self.backbone.eval()
