import glob
import random
import os

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

from auto_augment import rand_augment_transform

class CustomDownSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        self.labels = np.array(dataset.label)
        self.num_classes = len(np.unique(self.labels))

        num_each_class = []
        for i in range(self.num_classes):
            num_class_i = (self.labels==i).sum()
            num_each_class.append(num_class_i)
        self.num_each_class = np.array(num_each_class).min()

        self.num_samples = self.num_each_class * self.num_classes

    def __iter__(self):
        idxs = []
        for i in range(self.num_classes):
            idxs_i = np.where(self.labels==i)[0]
            idxs_i = np.random.choice(idxs_i, self.num_each_class, replace=False)
            idxs += idxs_i.tolist()
        
        random.shuffle(idxs)
        
        return (idx for idx in idxs)

    def __len__(self):
        return self.num_samples

# RAF-DB Dataset
class RafDataset(Dataset):
    def __init__(self, raf_path, phase, transform = None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.label = dataset.iloc[:, LABEL_COLUMN].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        
        self.file_paths = []
        # use raf aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path)
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label, idx

# AffectNet Dataset
class AffectDataset_7label(Dataset):
    def __init__(self, aff_path, phase, use_cache = True, transform = None):
        self.phase = phase
        self.transform = transform
        self.aff_path = aff_path
        
        if use_cache:
            cache_path = os.path.join(aff_path,'affectnet.csv')
            if os.path.exists(cache_path):
                df = pd.read_csv(cache_path)
            else:
                df = self.get_df()
                df.to_csv(cache_path)
        else:
            df = self.get_df()

        self.data = df[df['phase'] == phase]

        self.file_paths = self.data.loc[:, 'img_path'].values
        self.label = self.data.loc[:, 'label'].values

        self.file_paths = np.array(self.file_paths)
        self.label = np.array(self.label)
        idxs = np.where(self.label!=7)[0]
        self.file_paths = self.file_paths[idxs].tolist()
        self.label = self.label[idxs].tolist()

    def get_df(self):
        train_path = os.path.join(self.aff_path, 'train_set/')
        val_path = os.path.join(self.aff_path, 'val_set/')
        data = []
        
        for anno in glob.glob(train_path + 'annotations/*_exp.npy'):
            idx = os.path.basename(anno).split('_')[0]
            img_path = os.path.join(train_path, f'images/{idx}.jpg')
            label = int(np.load(anno))
            data.append(['train', img_path, label])
        
        for anno in glob.glob(val_path + 'annotations/*_exp.npy'):
            idx = os.path.basename(anno).split('_')[0]
            img_path = os.path.join(val_path, f'images/{idx}.jpg')
            label = int(np.load(anno))
            data.append(['val', img_path, label])
        
        return pd.DataFrame(data = data,columns = ['phase', 'img_path', 'label'])
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label, idx

# SFEW Dataset
class SFEWDataset(torchvision.datasets.ImageFolder):
    def __init__(self, sfew_path, phase, transform=None):
        if phase == 'train':
            super().__init__(os.path.join(sfew_path, 'Train'), transform=transform)
        else:
            super().__init__(os.path.join(sfew_path, 'Val'), transform=transform)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx

def get_dataloaders(dataset='raf', data_path='./datasets/raf-basic', batch_size=64, num_workers=2, num_samples=30000):
    # transforms 
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225) 
    if dataset in ['raf', 'affectnet7']:
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            rand_augment_transform(config_str='rand-m5-n3-mstd0.5', hparams={'translate_const': 117, 'img_mean': (124, 116, 104)}),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(scale=(0.02, 0.25)),
            ])
        data_transforms_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            ])
    elif dataset == 'sfew':
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),      
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.ToTensor()(rand_augment_transform(config_str='rand-m3-n5-mstd0.5',hparams={'translate_const': 117, 'img_mean': (124, 116, 104)})(crop)) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.RandomHorizontalFlip()(crop) for crop in crops])),        
            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.Normalize(mean=mean, std=std)(t) for t in tensors])),
            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.RandomErasing(scale=(0.02, 0.25))(t) for t in tensors])),
            ])
        data_transforms_val = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.Normalize(mean=mean, std=std)(t) for t in tensors])),
            ])

    # datasets
    if dataset == 'raf':
        dataset = RafDataset
    elif dataset == 'affectnet7':
        dataset = AffectDataset_7label
    elif dataset == 'sfew':
        dataset = SFEWDataset

    train_dataset = dataset(
        data_path,
        phase='train',
        transform=data_transforms
    )
    val_dataset = dataset(
        data_path,
        phase='test',
        transform=data_transforms_val
    )

    # dataloaders
    if dataset in [AffectDataset_7label]:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            persistent_workers=True,
            sampler=CustomDownSampler(train_dataset)
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True, 
            drop_last=True,
            persistent_workers=True,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=True
    )

    return train_loader, val_loader
