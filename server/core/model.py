import torch
import torch.nn as nn
from torchvision import models

class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        maxv = self.max_pool(x).view(b, c)
        out = self.mlp(avg) + self.mlp(maxv)
        return x * out.sigmoid().view(b, c, 1, 1)

class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        maxv, _ = x.max(dim=1, keepdim=True)
        x_cat = torch.cat([avg, maxv], dim=1)
        return x * self.conv(x_cat).sigmoid()

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_gate = ChannelGate(channels, reduction)
        self.spatial_gate = SpatialGate()

    def forward(self, x):
        x = self.channel_gate(x)
        x = self.spatial_gate(x)
        return x

class ResNet34_LDL_CBAM(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.resnet = models.resnet34(pretrained=False)
        self.backbone_fc_in = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.cbam = CBAM(512, reduction=16)
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        # ⚠️ NAMES MUST MATCH TRAINING
        self.classifier = nn.Linear(self.backbone_fc_in, num_classes)
        self.ldl_head   = nn.Linear(self.backbone_fc_in, num_classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.cbam(x)
        x = self.gap(x).view(x.size(0), -1)

        logits = self.classifier(x)
        dist_logits = self.ldl_head(x)

        return logits, dist_logits
