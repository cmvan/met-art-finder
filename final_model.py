import torch
import os
import pandas as pd
import numpy as np
from torch import nn, optim
from torchvision import models, transforms
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from PIL import Image 

class MuseumDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, mode):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }[mode]



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        title = self.data.iloc[idx]["title"]
        img_name = os.path.join(self.root_dir, title) + "..jpg"

        image = Image.open(img_name)
        # if image.mode == "L":
        #     image = Image.merge("RGB", (image, image, image))
        #     image.save(img_name)
        #     print("hi")
        # else:
        #     image = image.convert('RGB')
        #     image.save(img_name)

        label = self.data.iloc[idx]["label"]
        
        sample = {'image': image, 'label': label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
        sample['label'] = torch.tensor(label)
        # print(sample['image'].shape, )
        return sample


def train_model(model, device, dataloader, num_epochs=25, log_interval=100):
    for epoch in range(num_epochs):
        running_corrects = 0

        for batch_idx, sample in enumerate(dataloader):
            data = sample["image"]
            targets = sample["label"]
            data, targets = data.to(device), targets.to(device)
            
            

            # Zero the gradients
            optimizer.zero_grad()
            
            # Enable autocasting for mixed precision
            with torch.amp.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
                # Forward pass
                outputs = model(data)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, targets)
            
            # Perform backward pass and gradient scaling
            scaler.scale(loss).backward()
            
            # Update model parameters
            scaler.step(optimizer)
            scaler.update()
            
            running_corrects += torch.sum(preds == targets.data)

            # Print training progress
            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        epoch_acc = running_corrects.double() / 1709
        print(f"Epoch {epoch+1} Acc: {epoch_acc:.4f}")


if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = models.vgg19(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_features, 31)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda")


    transformed_dataset = MuseumDataset(csv_file='final_data/merged_labels.csv',
                                        root_dir='final_data/merged_museum_images',
                                        mode='train')
                                           
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=0)
    

    train_model(model, device, dataloader)

    torch.save(model.state_dict(), "new_finetuned.pth")


    