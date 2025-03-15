import torch.nn as nn
from torchvision import models
from torchvision.models import VGG19_Weights

if __name__ == "__main__":
    vgg19_model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

    for param in vgg19_model.parameters():  # get all the parameters of the model
        param.requires_grad = False  # these layers won't be update during training

    # Unfreeze weights of last two fully connected layers (FC1 and FC2)
    for param in vgg19_model.classifier[0].parameters():
        param.requires_grad = True  # will be updated during training
    for param in vgg19_model.classifier[3].parameters():
        param.requires_grad = True  # will be updated during training



    # (Recommended) Modify the last layer for your number of classes
    # class_to_idx = TODO
    # num_classes = len(class_to_idx)
    # model.classifier[6] = nn.Linear(4096, num_classes)