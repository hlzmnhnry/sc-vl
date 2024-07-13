import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from itertools import chain
from torchvision import models
from collections import OrderedDict
from utils.helpers import initialize_weights, set_trainable

def load_custom_resnet(model, backbone, extension, pretrained_dir="pretrained"):

    state_dict = torch.load(os.path.join(pretrained_dir, f"{backbone}_{extension}.pth"))
    new_state_dict = OrderedDict()

    if extension == "gray":

        state_dict = state_dict["model"]

        renamings = {
            "conv1.weight": "layer0.0.weight",
            "bn1.weight": "layer0.1.weight",
            "bn1.bias": "layer0.1.bias",
            "bn1.running_var": "layer0.1.running_var",
            "bn1.running_mean": "layer0.1.running_mean",
            "bn1.num_batches_tracked": "layer0.1.num_batches_tracked"
        }

        ignore_keys = ["fc.weight", "fc.bias"]

        for k, v in state_dict.items():

            if k not in ignore_keys:

                name = k

                if name in renamings.keys():
                    name = renamings[name]

                new_state_dict[name] = v

        state_dict = new_state_dict

    elif extension == "coco_gray":

        for k, v in state_dict.items():

            name = k.replace("module.", "")
            
            if name.split(".")[0] == "backbone":
                
                name = name.replace("backbone.", "")
                new_state_dict[name] = v
            
        state_dict = new_state_dict

    model.load_state_dict(state_dict)

def load_custom_xception(model, backbone, extension, pretrained_dir="pretrained"):

    state_dict = torch.load(os.path.join(pretrained_dir, f"{backbone}_{extension}.pth"),
        map_location=torch.device("cpu"))

    if extension == "gray":

        state_dict = state_dict["model"]

        if "last_linear.weight" in state_dict:
            del state_dict["last_linear.weight"]

        if "last_linear.bias" in state_dict:
            del state_dict["last_linear.bias"]

    elif extension == "coco_gray":
        
        new_state_dict = OrderedDict()

        if isinstance(state_dict, dict) and "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]

        for k, v in state_dict.items():

            name = k.replace("module.", "")
            
            if name.split(".")[0] == "backbone":
                
                name = name.replace("backbone.", "")
                new_state_dict[name] = v
            
        state_dict = new_state_dict

    model.load_state_dict(state_dict)

class ResNet(nn.Module):

    def __init__(self, in_channels=1, output_stride=16, backbone="resnet101", backbone_extension=None,
        pretrained=True):

        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained and in_channels == 3)

        if not pretrained or in_channels != 3:

            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

            initialize_weights(self.layer0)

        else:
            self.layer0 = nn.Sequential(*list(model.children())[:4])

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if output_stride == 16: s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8: s3, s4, d3, d4 = (1, 1, 2, 4)
        
        if output_stride == 8:

            for n, m in self.layer3.named_modules():

                if "conv1" in n and (backbone == "resnet34" or backbone == "resnet18"):
                    m.dilation, m.padding, m.stride = (d3,d3), (d3,d3), (s3,s3)
                elif "conv2" in n:
                    m.dilation, m.padding, m.stride = (d3,d3), (d3,d3), (s3,s3)
                elif "downsample.0" in n:
                    m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():

            if "conv1" in n and (backbone == "resnet34" or backbone == "resnet18"):
                m.dilation, m.padding, m.stride = (d4,d4), (d4,d4), (s4,s4)
            elif "conv2" in n:
                m.dilation, m.padding, m.stride = (d4,d4), (d4,d4), (s4,s4)
            elif "downsample.0" in n:
                m.stride = (s4, s4)

        if pretrained and backbone_extension in ["gray", "coco_gray"]:
            load_custom_resnet(self, backbone, backbone_extension)

    def forward(self, x):

        x = self.layer0(x)
        x = self.layer1(x)

        low_level_features = x

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_features

class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=0, dilation=1, bias=False):

        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation,
            groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):

        x = self.conv1(x)
        x = self.pointwise(x)

        return x

class Block(nn.Module):

    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        rep = []

        filters=in_filters

        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep)

    def forward(self,inp):

        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip

        return x

class Xception(nn.Module):

    def __init__(self, in_channels=1, backbone="xception", backbone_extension=None,
        pretrained=True):

        super(Xception, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        if pretrained and backbone_extension in ["gray", "coco_gray"]:

            load_custom_xception(self, backbone, backbone_extension)

        else:

            for m in self.modules():

                if isinstance(m, nn.Conv2d):

                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

                elif isinstance(m, nn.BatchNorm2d):

                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def features(self, input):

        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.block1(x)

        low_level_features = x

        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)

        return x, low_level_features

    def logits(self, features):

        x = nn.ReLU(inplace=True)(features)

        return x

    def forward(self, input):

        x, low_level_features = self.features(input)
        x = self.logits(x)

        return x, low_level_features

def assp_branch(in_channels, out_channles, kernel_size, dilation):

    padding = 0 if kernel_size == 1 else dilation

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channles),
        nn.ReLU(inplace=True)
    )

class ASSP(nn.Module):

    def __init__(self, in_channels, output_stride):
    
        super(ASSP, self).__init__()

        assert output_stride in [8, 16], "Only output strides of 8 or 16 are suported"
        if output_stride == 16: dilations = [1, 6, 12, 18]
        elif output_stride == 8: dilations = [1, 12, 24, 36]
        
        self.aspp1 = assp_branch(in_channels, 256, 1, dilation=dilations[0])
        self.aspp2 = assp_branch(in_channels, 256, 3, dilation=dilations[1])
        self.aspp3 = assp_branch(in_channels, 256, 3, dilation=dilations[2])
        self.aspp4 = assp_branch(in_channels, 256, 3, dilation=dilations[3])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv1 = nn.Conv2d(256*5, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        initialize_weights(self)

    def forward(self, x):

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)),
            mode="bilinear", align_corners=True)

        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))

        return x

class Decoder(nn.Module):

    def __init__(self, low_level_channels, num_classes):

        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)

        self.output = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1, stride=1)
        )

        initialize_weights(self)

    def forward(self, x, low_level_features):

        low_level_features = self.conv1(low_level_features)
        low_level_features = self.relu(self.bn1(low_level_features))
        H, W = low_level_features.size(2), low_level_features.size(3)

        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)
        x = self.output(torch.cat((low_level_features, x), dim=1))

        return x

class DeepLab(BaseModel):

    def __init__(self, num_classes, in_channels=1, backbone="xception", backbone_extension=None,
        pretrained=True, output_stride=16, freeze_bn=False, freeze_backbone=False, **_):
                
        super(DeepLab, self).__init__()
        assp_in = 2048
        
        assert ("xception" or "resnet" in backbone)

        if "resnet" in backbone:

            self.backbone = ResNet(backbone=backbone, in_channels=in_channels,
                output_stride=output_stride, backbone_extension=backbone_extension,
                pretrained=pretrained)
            low_level_channels = 256 if int(backbone[6:]) in [50, 101, 152] else 64
            if int(backbone[6:]) in [18, 34]: assp_in = 512
            
        else:

            self.backbone = Xception(in_channels=in_channels, backbone=backbone,
                backbone_extension=backbone_extension, pretrained=pretrained)
            low_level_channels = 128

        self.ASSP = ASSP(in_channels=assp_in, output_stride=output_stride)
        self.decoder = Decoder(low_level_channels, num_classes)

        if freeze_bn: self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.backbone], False)

    def forward(self, x):

        H, W = x.size(2), x.size(3)

        x, low_level_features = self.backbone(x)
        x = self.ASSP(x)
        x = self.decoder(x, low_level_features)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)

        return x

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.ASSP.parameters(), self.decoder.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
