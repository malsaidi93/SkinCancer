import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F

import torch.nn as nn
import torch.nn.functional as F

class CNN2DModel(nn.Module):
    def __init__(self, num_classes):
        super(CNN2DModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 192, 3, padding=1)
        self.conv3 = nn.Conv2d(192, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.local_response_norm(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.local_response_norm(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def efficientnet():
    model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
    old_fc = model.classifier.__getitem__(-1)
    new_fc = nn.Linear(in_features=old_fc.in_features, out_features= 7, bias=True)
    model.classifier.__setitem__(-1 , new_fc)
    return model

def resnet(output):
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    old_fc = model.fc
    new_fc = nn.Linear(in_features=old_fc.in_features, out_features= 100, bias=True)
    model.fc = new_fc
    
    return model

def vit():
    model = models.vit_b_16(weights='ViT_B_16_Weights.DEFAULT')
    
    old_fc = model.heads.__getitem__(-1)
    new_fc = nn.Linear(in_features=old_fc.in_features, out_features= 7, bias=True)
    model.heads.__setitem__(-1 , new_fc)

    return model


def convnext():
    model = models.convnext_base(weights='ConvNeXt_Base_Weights.DEFAULT')
    old_fc = model.classifier.__getitem__(-1)
    new_fc = nn.Linear(in_features=old_fc.in_features, out_features= 7, bias=True)
    model.classifier.__setitem__(-1 , new_fc)

    return model

def alexnet():
    model = models.alexnet(pretrained=True)
    old_fc = model.classifier.__getitem__(-1)
    new_fc = nn.Linear(in_features=old_fc.in_features, out_features= 7, bias=True)
    model.classifier.__setitem__(-1 , new_fc)
    
    return model


def resnext():
    # weights can without quotes if there is error
    model = models.resnext50_32x4d(weights='ResNeXt50_32X4D_Weights.DEFAULT')
    old_fc = model.fc
    new_fc = nn.Linear(in_features=old_fc.in_features, out_features= 7, bias=True)
    model.fc = new_fc
    
    return model
