import os, sys, glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from PIL import Image, ImageOps
import random
import json
from sklearn.utils import class_weight
from scipy.special import softmax
from torch.utils.data import ConcatDataset
# from torchvision.transforms import v2
# from do_augmentation import augment

class SkinCancerCustom(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label


class CombinedDataset(Dataset):
    def __init__(self, csv_dataset, array_dataset):
        self.csv_dataset = csv_dataset
        self.array_dataset = array_dataset

    def __len__(self):
        return len(self.csv_dataset) + len(self.array_dataset)

    def __getitem__(self, index):
        if index < len(self.csv_dataset):
            return self.csv_dataset[index]
        else:
            return self.array_dataset[index - len(self.csv_dataset)]

class SkinCancer(Dataset):
    """Skin Cancer Dataset."""

    def __init__(self, root_dir, meta, transform=None, augment_phase=False, classes_to_augment=[]):
        """
        Args:
            root_dir (string): Path to root directory containing images
            meta_file (string): Path to csv file containing images metadata (image_id, class)

            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.landmarks_frame = pd.read_csv(csv_file)
        
        self.root_dir = root_dir
        self.meta = meta
        self.transform = transform

        self.df = pd.read_csv(self.meta)
        try:
            self.image_paths = self.df['image_pth'].to_list()
        except:
            self.image_paths = self.df['image_path'].to_list()
            
        self.image_ids = self.df['image_id'].to_list()
        self.classes = sorted(self.df['dx'].unique().tolist())
        self.classes_all = self.df['dx'].tolist()

        self.class_id = {i:j for i, j in enumerate(self.classes)}
        self.class_to_id = {value:key for key,value in self.class_id.items()}

        self.class_count =  self.df['dx'].value_counts().to_dict()
        self.transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            # transforms.RandomHorizontalFlip(),  # Random horizontal flip
                                            # transforms.RandomVerticalFlip(),  # Random vertical flip
                                            # transforms.RandomRotation(20),
                                            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
                                            # transforms.RandomGrayscale(p=0.5),  # Randomly convert to grayscale
                                            # transforms.RandomInvert(0.5),
                                            # transforms.RandomAutocontrast(0.5),
                                            transforms.RandomAdjustSharpness(2,0.5),
                                            transforms.Resize((224,224)),
                                            transforms.Normalize([0.5], [0.5])
                                            ])
        self.transform_NoAug = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224)), transforms.Normalize([0.5], [0.5])])
        self.augment_phase = augment_phase
        self.classes_to_augment = classes_to_augment
        

        # self.class_weights = [1 - self.class_count[i]/self.df.shape[0] for i in self.classes]
        # self.class_weights = torch.tensor(class_weight.compute_class_weight('balanced',classes=np.unique(self.df['dx'].to_numpy()),y=self.df['dx'].to_numpy()),device='cuda')
        # self.file_names_ids = {i:v for v,i in enumerate(self.file_names)}

    def __len__(self):
        return len(self.image_paths)
    
    def __distribution__(self):
            return dict(self.df['dx'].value_counts())
    

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, -1]
        label = self.df.iloc[idx, 2]
        image = Image.open(img_path)
        if self.augment_phase and label in self.classes_to_augment:
            image_tensor = self.transform(image)
        else:
            image_tensor = self.transform_NoAug(image)
        label_id = torch.tensor(self.class_to_id[str(label)])
        return image_tensor, label_id

class SkinCancerWithAugmentation(Dataset):
    """Skin Cancer Dataset."""

    def __init__(self, root_dir, meta, transform=None, augment_phase=False, classes_to_augment=None):
        """
        Args:
            root_dir (string): Path to root directory containing images
            meta_file (string): Path to csv file containing images metadata (image_id, class)

            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.landmarks_frame = pd.read_csv(csv_file)
        
        self.root_dir = root_dir
        self.meta = meta
        self.transform = transform

        self.df = pd.read_csv(self.meta)
        try:
            self.image_paths = self.df['image_pth'].to_list()
        except:
            self.image_paths = self.df['image_path'].to_list()
            
        self.image_ids = self.df['image_id'].to_list()
        self.classes = sorted(self.df['dx'].unique().tolist())
        self.classes_all = self.df['dx'].tolist()

        self.class_id = {i:j for i, j in enumerate(self.classes)}
        self.class_to_id = {value:key for key,value in self.class_id.items()}

        self.class_count =  self.df['dx'].value_counts().to_dict()
        self.transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.augment_phase = augment_phase
        self.classes_to_augment = classes_to_augment
        self.softmax = pd.read_excel('../reports/combined_reports.xlsx', sheet_name='Softmax_Values', index_col=0)
        self.aug_list = self.__augmentationslist__(self.softmax)

    def __getclassificationreport__(self):
        
        combined_reports = {}
        root = "../reports/" 
        for file in os.listdir(root):
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    report = json.load(f)
                combined_reports[file] = {each_class: report[each_class]['f1-score'] for each_class in report if each_class in self.classes}
        
        forSoftmax = []
        aug = []                                                                      
        for filename, values in combined_reports.items():                             
            row = [filename]                                                          
            row.extend(values.values())                                               
            aug.append(filename)                                                      
            forSoftmax.append(list(values.values()))
        
        # Calculate softmax across the columns (axis=0)
        softmax_data = softmax(forSoftmax, axis=0)
        
        # create Dictionary
        softmax_dict = {}
        for idx, probs in enumerate(softmax_data):
            softmax_dict[probs]
            row.extend(probs)                  


    def __len__(self):
        return len(self.image_paths)

    def __distribution__(self):
            return dict(self.df['dx'].value_counts())
            
    def __augmentationslist__(self, df, threshold=0.20):
        augmentations_dict = {}
        
        # Iterate over columns (classes)
        for class_name in df.columns:
            transform_list = []
            
            # Iterate over rows (augmentations)
            for augmentation_name, probability in df[class_name].items():
                if probability > threshold:
                    if 'RandomVerticalFlip' in augmentation_name:
                        transform_list.append(transforms.RandomVerticalFlip())
                    elif 'RandomHorizontalFlip' in augmentation_name:
                        transform_list.append(transforms.RandomHorizontalFlip())
                    elif 'RandomGrayScale' in augmentation_name:
                        transform_list.append(transforms.RandomGrayscale())
                    elif 'RandomColorJitter' in augmentation_name:
                        transform_list.append(transforms.ColorJitter())
                    elif 'RandomRotation' in augmentation_name:
                        transform_list.append(transforms.RandomRotation(30))
                    elif 'RandomInvert' in augmentation_name:
                        transform_list.append(transforms.RandomInvert(0.5))
                    elif 'RandomAdjustSharpness' in augmentation_name:
                        transform_list.append(transforms.RandomAdjustSharpness(2, 0.5))
                    elif 'RandomContrast' in augmentation_name:
                        transform_list.append(transforms.RandomContrast(0.5))
            # Combine the selected transforms
            augmentations_dict[class_name] = transforms.Compose(transform_list)
        # print(augmentations_dict)
        return augmentations_dict
    
    def __getitem__(self, idx):
        
        img_path = self.df.iloc[idx, -1]
        label = self.df.iloc[idx, 2]
        image = Image.open(img_path)
        
        # if label in self.classes_to_augment:
        #     image_tensor = self.transforms(image)
        # else:
        #     image_tensor = self.transforms.ToTensor()(image)
        image_tensor = self.transform(image)
        image_tensor = self.aug_list[label](image_tensor)
        label_id = torch.tensor(self.class_to_id[str(label)])
        return image_tensor, label_id



