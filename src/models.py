import torch
import torch.nn as nn
from torchvision import models


class cnn():
    pass


def efficientnet():

    model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
    old_fc = model.classifier.__getitem__(-1)
    new_fc = nn.Linear(in_features=old_fc.in_features, out_features= 7, bias=True)
    model.classifier.__setitem__(-1 , new_fc)

    # model.classifier[1].out_features = 7

    return model



def resnet():
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    # model.fc.out_features = 7
    old_fc = model.fc
    new_fc = nn.Linear(in_features=old_fc.in_features, out_features= 7, bias=True)
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
    
    # model.classifier[2].out_features = 7

    return model



