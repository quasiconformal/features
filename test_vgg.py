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

#print(torch.__version__)

if __name__ == '__main__':

    vgg19 = models.vgg19(pretrained=True)

    img_org = Image.open('data/baboon.jpg')

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = preprocess(img_org)
    print(type(img))
    img_batch = img[None]
    vgg19.eval()

    result = vgg19(img_batch)
    print(type(result))
    print(result.shape)

    idx = torch.argmax(result[0])
    print(idx)
    # tensor(372)

    print(idx.ndim)
    # 0

    with open('./data/imagenet_class_index.json') as f:
        labels = json.load(f)

    print(type(labels))
    # <class 'dict'>

    print(len(labels))
    # 1000

    print(labels['0'])
    # ['n01440764', 'tench']

    print(labels['999'])
    # ['n15075141', 'toilet_tissue']

    print(labels[str(idx.item())])
    # ['n02486410', 'baboon']

    probabilities = torch.nn.functional.softmax(result, dim=1)[0]
    print(probabilities.shape)
    # torch.Size([1000])

    print(probabilities.sum())
    # tensor(1.0000, grad_fn=<SumBackward0>)

    print(probabilities[idx])
    # tensor(0.5274, grad_fn=<SelectBackward>)

    print(probabilities[idx.item()])
    # tensor(0.5274, grad_fn=<SelectBackward>)

    _, indices = torch.sort(result[0], descending=True)
    print(indices.shape)
    # torch.Size([1000])

    for idx in indices[:5]:
        print(labels[str(idx.item())][1], ':', probabilities[idx].item())