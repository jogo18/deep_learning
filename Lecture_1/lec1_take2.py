# The Kaggle Python 3 environment comes with many helpful tools installed. 
# See the kaggle/python Docker image: https://github.com/kaggle/docker-python
# If using another environment you may need to install the packages yourself
# e.g. pip install pytorch_lightning or !pip install pytorch_lightning

import torch.nn.functional as F
import numpy as np 
import pandas as pd 
import os, sys 
import matplotlib.pyplot as plt
import cv2
import random
import torch
import wandb
import pytorch_lightning as pl
import torchvision.transforms as transforms
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from argparse import Namespace
from pytorch_lightning.loggers import WandbLogger
from argparse import Namespace

class LightningClassifier(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()

        self.lr = hparams.learning_rate
        self.model = ResNet18(image_channels=hparams.image_channels, num_classes=hparams.n_outputs)
        self.loss_function = nn.CrossEntropyLoss() #nn.BCEWithLogitsLoss()
        self.softmax = nn.Softmax(dim=1)
        from torchmetrics import Accuracy, Precision, Recall, F1Score, MetricCollection
        self.metric_collection = MetricCollection([F1Score(task='multiclass', num_classes=hparams.n_outputs, average='macro'), # https://torchmetrics.readthedocs.io/en/stable/classification/f1_score.html
                                                Accuracy(task='multiclass', num_classes=hparams.n_outputs, average='macro'), # https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html
                                                ])
        self.valid_metrics = self.metric_collection.clone(prefix='val_')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_ = self.forward(x)
        loss = self.loss_function(y_, y)
        self.log('loss', loss, on_step=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_ = self.forward(x)
        loss = self.loss_function(y_, y)
        scores = self.softmax(y_)
        metrics = self.valid_metrics(scores, y)
        self.log_dict(metrics)
        return {'loss': loss}
class ResNet18(nn.Module):
    
    def __init__(self, image_channels, num_classes):
        
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def __make_layer(self, in_channels, out_channels, stride):
        
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
            
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride), 
            Block(out_channels, out_channels)
        )
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x 
        
    def identity_downsample(self, in_channels, out_channels):
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(out_channels)
        )
class Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x
class AgeClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, labels, transforms=None):
        # Make sure that image_list and labels are not unequal length
        if len(image_list) != len(labels):
            raise ValueError('image_list and labels are not equal length.')
        
        self.image_list = image_list
        self.labels = labels
        self.arranged_categories = ['YOUNG', 'MIDDLE', 'OLD']
        self.transforms = transforms

    def load_sample(self, idx):
        img = Image.open(self.image_list[idx])
        label = torch.as_tensor(self.arranged_categories.index(self.labels[idx]), dtype=torch.int64) # convert categorical string label to int and the to one hot
        #label = F.one_hot(torch.as_tensor(self.arranged_categories.index(self.labels[idx]), dtype=torch.int64), num_classes=3) # convert categorical string label to int and the to one hot
        return img, label

    def __getitem__(self, idx):
        img, target = self.load_sample(idx)

        if self.transforms:
            img = self.transforms(img)
        else:
            convert_tensor = transforms.ToTensor()
            img = convert_tensor(img)

        return img, target#torch.unsqueeze(target,0)

    def __len__(self):
        return len(self.labels)
def main():        
    # When using Kaggle datasets upload of data is handled behind the scene
    # Data files are available in the read-only "../input/" directory
    # If you are using something else, you need to upload the data manually
    # Here we take a look at the content of the input directory
    input_dir = '/home/jagrole/AAU/8.Semester/deep_learning/Data/Lec1/faces' 
    print(f"input dir contains: {os.listdir(input_dir)}")
    # dataset_dir = os.path.join(input_dir,'faces-age-detection-dataset')
    print(f"dataset dir contains: {os.listdir(input_dir)}")
    train_dir = os.path.join(input_dir,'Train')
    print(f"train dir contains: {len(os.listdir(train_dir))} files such as {os.listdir(train_dir)[0]}")


    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        print("using {}".format(torch.cuda.get_device_name(0)))
    else:
        print("using CPU")


    # Lets see what is in 'train.csv'
    train_df = pd.read_csv(os.path.join(input_dir,'train.csv'))
    print(f'{train_df.head()}\n')

    # Get an overview of the labels
    labels = train_df['Class'].values
    values, counts = np.unique(labels, return_counts=True) # count instances for each unique label
    print(f'values {values}')
    print(f'counts {counts}')

    # Inspect a couple of images
    image_list = [os.path.join(train_dir,f) for f in train_df['ID'].values] # include the full path in the list
    for _ in range(2):
        idx = random.randint(0,len(train_df))
        image = cv2.imread(image_list[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f'Image {image_list[idx]} with shape {image.shape} and label {labels[idx]}')
        plt.imshow(image)
        plt.show()

    
    dataset = AgeClassificationDataset(image_list=image_list, labels=labels, transforms=None) 
    print(dataset[0])
    # from here: https://www.kaggle.com/code/ivankunyankin/resnet18-from-scratch-using-pytorch
        
    # Here we define the model, optimizer, train and validation loop


    hparams = Namespace(**{# data
                        'exp_name': 'mp_age_classifier', # unique name for model and logs
                        'image_size': 224, # images size for input to model
                        'image_channels': 3, # 3 for RGB, 1 for grayscale images
                        # model
                        'arch': 'resnet18', # 'resnet18',
                        'n_outputs': 3, # number of dependent variables
                        # training
                        'wandb_api_key': '89a8a6910a20129d4e08086de0cfef60afe818c5',
                        'wandb_project': 'my-awesome-project',
                        'gpus': 1, # number of gpus
                        'max_epochs': 5, # number of times during training, where the whole dataset is traversed
                        'learning_rate': 1e-4, # generally, high means faster but worse convergence, low slow but better convergence
                        'batch_size': 32, # should be considered together with learning rate. decrease if using a small machine and getting memory errors
                        'n_workers': 2 # set to 0 in windows when working with a windows on a small machine
                        }
                        )

    model_name=f"{hparams.exp_name}_{hparams.arch}_{hparams.image_size}x{hparams.image_channels}-{hparams.n_outputs}"
    print("Training model: {}".format(model_name))

    # prepare output directories
    if not os.path.exists('logs/'):
        os.mkdir('logs/')
    output_dir = os.path.join('trained_models/',model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    os.environ["WANDB_API_KEY"] = hparams.wandb_api_key
    wandb.init(config=vars(hparams))
    hparams = Namespace(**wandb.config)

    logger = WandbLogger(save_dir='wandb_logs/', offline=False, project=hparams.wandb_project, log_model=False)

    # The network needs the input to be a specific size and to pytorch tensors 
    train_transforms = transforms.Compose([transforms.Resize([hparams.image_size,hparams.image_size]),
                                        transforms.ToTensor()])

    train_val_set = AgeClassificationDataset(image_list=image_list, labels=labels, transforms=train_transforms) 
    # since we only have the Train folder, we split its contents into training and validation set
    proportions = [0.8, 0.2]
    lengths = [int(p * len(train_val_set)) for p in proportions]
    lengths[-1] = len(train_val_set) - sum(lengths[:-1])
    train_set, val_set = torch.utils.data.random_split(train_val_set, lengths)
    print(f"train_val_set of len {len(train_val_set)} split into train_set {len(train_set)} and val_set {len(val_set)}")

    train_dataloader = DataLoader(train_set, batch_size=hparams.batch_size, shuffle=True, pin_memory=True, num_workers=hparams.n_workers, persistent_workers=True)
    val_dataloader = DataLoader(val_set, batch_size=hparams.batch_size, shuffle=False, pin_memory=True, num_workers=hparams.n_workers, persistent_workers=True)


    # initialize model, training and validation code
    lightning_module = LightningClassifier(hparams)

    # initialize training
    trainer = pl.Trainer(accelerator='gpu', 
                        max_epochs=hparams.max_epochs,
                        logger=logger,
                        )
    trainer.fit(lightning_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


    predictor = lightning_module.model.eval().cpu()
    test_set = val_set

    labels, predictions = [], []
    for i in range(len(test_set)):
        img, label = test_set[i]
        labels.append(label.numpy())
        with torch.no_grad():
            pred = predictor(torch.unsqueeze(img, 0))[0].cpu().numpy()
            predictions.append(np.argmax(pred))
        if i > 500:
            break

    cm = confusion_matrix(labels, predictions, labels=[1,2,3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,2,3])
    disp.plot()
    plt.show()


if __name__ == '__main__':
    main()