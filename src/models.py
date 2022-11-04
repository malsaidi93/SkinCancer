import torch
import torch.nn as nn
from torchvision import models


class CNN():
    pass


def efficientnet():

    model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
    model.classifier[1].out_features = 7

    return model

def vgg():

    model = models.vgg19(weights='VGG19_Weights.DEFAULT')
    model.classifier[6].out_features = 7

    return model


def resnet():
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    model.fc.out_features = 7
    return model


def vit():
    model = models.vit_b_16(weights='ViT_B_16_Weights.DEFAULT')
    
    model.heads[0].out_features = 7

    return model


def convnext():
    model = models.convnext_base(weights='ConvNeXt_Base_Weights.DEFAULT')
    model.classifier[2].out_features = 7

    return model



