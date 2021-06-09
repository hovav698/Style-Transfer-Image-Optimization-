import torch
import numpy as np
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load and transform the image
def load_img(path_to_img):
    max_dim = 512
    img = Image.open(path_to_img)
    img_arr = np.array(img)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),

    ])

    img = transform(img)
    img = torch.unsqueeze(img, 0)

    return img


def gram_matrix(input):
    # calculation of the gram matrix for the input.
    # The inputs will be the the generated image and the style image
    batch_size, channel, height, width = input.shape
    G = torch.mm(input.view(channel, height * width), input.view(channel, height * width).t())
    return G / (height * width)
