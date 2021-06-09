import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import IPython.display as display
import time
from torchvision import transforms
import cv2
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





# create the VGG model that will be used for cost function calculation. the above conv layers will be used as output
class vgg19(torch.nn.Module):
    def __init__(self, content_layers_idx, style_layers_idx):
        super(vgg19, self).__init__()

        self.content_layers_idx = content_layers_idx
        self.style_layers_idx = style_layers_idx

        self.vgg = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
        # we will not change the model weights. will only change the image pixels
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, img):
        # we will use different output layers for the content loss and the style loss calculation
        content_outputs = []
        style_outputs = []
        features = list(self.vgg.features)
        out = img
        features = torch.nn.ModuleList(features).eval()

        for i, model in enumerate(features):
            out = model(out)
            if i in self.content_layers_idx:
                content_outputs.append(out)
            if i in self.style_layers_idx:
                style_outputs.append(out)

        return content_outputs, style_outputs