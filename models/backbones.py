import torch
import torchvision.models as tv_models


class ResNet50Backbone(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = tv_models.resnet50(weights=weights)

        self.stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.out_channels = (512, 1024, 2048)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        p3 = self.layer2(x)
        p4 = self.layer3(p3)
        p5 = self.layer4(p4)
        return p3, p4, p5
