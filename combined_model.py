import pandas as pd
import numpy as np
import os
from PIL import Image

import torch
print(torch.__version__)
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

data_path = "./final_data/"

class ImageDataset(Dataset):
    """Tabular and Image dataset."""

    def __init__(self, csv_file, image_dir, mode):
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_file)
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

        tabular = self.data.iloc[idx, 0:]

        label = tabular["label"]

        title = tabular["title"]
        img_name = os.path.join(self.root_dir, title) + "..jpg"
        image = Image.open(img_name)

        tabular = tabular[["title", "artist", "medium", "department", "culture", "period", "classification"]]

        tabular = tabular.tolist()
        tabular = torch.FloatTensor(tabular)

        sample = {'image': image, 'tabular': tabular, 'label': label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
        sample['label'] = torch.tensor(label)

        return sample
    

class MetadataMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super(MetadataMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.output = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.output(x)
    

class MuseumClassifier(pl.LightningModule):
    def __init__(
        self, device, image_state, tab_in_dim=7, out_dim=31, lr: float = 1e-3, num_workers: int = 4, batch_size: int = 32
    ):
        super().__init__()
        self.lr = lr
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.tabular_model = MetadataMLP(tab_in_dim)
        self.tabular_model.to(device)
        # self.tabular_model.load_state_dict(torch.load(tabular_state, weights_only=True))

        self.image_model = models.vgg19(pretrained=True)
        num_features = self.image_model.classifier[-1].in_features
        self.image_model.classifier[-1] = nn.Linear(num_features, out_dim)
        self.image_model.load_state_dict(torch.load(image_state, weights_only=True))


        self.relu = nn.ReLU()

        self.ln1 = nn.Linear(self.tabular_model.output.out_features, out_dim)
        self.ln2 = nn.Linear(out_dim*2, out_dim) #figure out dims

    def forward(self, img, tab):
        img = self.image_model(img)
        tab = self.tabular_model(tab)
        tab = self.ln1(tab)

        x = torch.cat((img, tab), dim=1)
        x = self.relu(x)

        return self.ln2(x)
    
    def training_step(self, batch, batch_idx):
        image, tabular, y = batch

        criterion = nn.CrossEntropyLoss()
        y_pred = torch.flatten(self.forward(image, tabular))
        y_pred = y_pred.double()

        loss = criterion(y_pred, y)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        image, tabular, y = batch

        criterion = nn.CrossEntropyLoss()
        y_pred = torch.flatten(self.forward(image, tabular))
        y_pred = y_pred.double()

        val_loss = criterion(y_pred, y)

        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}
    
    def test_step(self, batch, batch_idx):
        image, tabular, y = batch

        criterion = nn.CrossEntropyLoss()
        y_pred = torch.flatten(self.forward(image, tabular))
        y_pred = y_pred.double()

        test_loss = criterion(y_pred, y)

        return {"test_loss": test_loss}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        logs = {"test_loss": avg_loss}
        return {"test_loss": avg_loss, "log": logs, "progress_bar": logs}

    def setup(self, stage):
        image_data = ImageDataset(csv_file=f"{data_path}merged_numerical_labels.csv", image_dir=f"{data_path}merged_museum_images/")

        train_size = int(0.80 * len(image_data))
        val_size = int((len(image_data) - train_size) / 2)
        test_size = int((len(image_data) - train_size) / 2)

        self.train_set, self.val_set, self.test_set = random_split(image_data, (train_size, val_size, test_size))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=(self.lr))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)


if __name__ == "__main__":
    device = "cuda:0"

    print(device)

    logger = TensorBoardLogger("lightning_logs", name="multi_input")
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=5000, patience=7, verbose=False, mode="min")

    model = MuseumClassifier(device, "./new_finetuned.pth")
    trainer = pl.Trainer(gpus=1, logger=logger, early_stop_callback=early_stop_callback)

    lr_finder = trainer.lr_find(model)
    fig = lr_finder.plot(suggest=True, show=True)
    new_lr = lr_finder.suggestion()
    model.hparams.lr = new_lr

    trainer.fit(model)
    trainer.test(model)