import matplotlib.pyplot as plt
import torchvision 
from torchvision import transforms
from PIL import Image
import numpy as np
import os 

# Loads a single image
def load_image(filename):
    img = Image.open(filename)
    return img


# Preprocesses image and converts it to a Tensor. Resizes image if there is a specified size.
def image_to_tensor(image,device,resize=None):

    if resize is not None:
        if type(resize) is not tuple:
            resize = (resize,resize)
        image = transforms.Resize(resize)(image)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    tensor = transform(image)
    return tensor.to(device)


# Converts tensor to an image
def tensor_to_image(tensor):
    postprocessor = transforms.Compose([
        transforms.ToPILImage(),
    ])
    image = postprocessor(tensor)
    return image