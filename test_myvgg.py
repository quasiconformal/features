import torch
from torchvision import models, transforms
from PIL import Image
import json

import numpy as np

import cv2
import math
import numpy as np
import random
import torch

import myvgg

if __name__ == '__main__':

    net = myvgg.myvgg19()
    #print(net)

    img_org = Image.open('data/cell.png')
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    net.eval()

    img = preprocess(img_org)
    img_batch = img[None]
    result = net(img_batch)
    print(type(result))
    print(result.shape)

    idx = torch.argmax(result[0])
    print(idx)