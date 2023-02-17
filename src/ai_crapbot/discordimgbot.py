import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

# Load the pre-trained ResNet-18 model
from torchvision.models import ResNet18_Weights

def returnimage(filename):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()

# Define the image transformation pipeline
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# Load the ImageNet class labels
    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

# Iterate over the images in the directory and classify them
    images = './images'
    filename = str(filename)
    image_path = os.path.join(images, filename)
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_name = classes[predicted.item()]
        return class_name
