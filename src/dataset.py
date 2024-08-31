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
                                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
                                            # transforms.RandomGrayscale(p=0.5),  # Randomly convert to grayscale
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
    
    # def __distribution__(self):
    #     # # data_dir = self.root_dir + '/train'
    #     # classes_path = glob.glob(self.root_dir+'/**/*.jpg')
    #     # # classes_path = glob.glob('../../skin_cancer_data/Train'+'/*')
    #     # classes = [i.split('/')[-1] for i in classes_path]
    #     class_dcit_lists=[]
    #     for idx in range(0, len(self.classes)):
    #         class_dict = {}
    #         class_dict['class'] = self.classes[idx]
    #         class_dict['files'] = glob.glob(self.root_dir+'/'+self.classes[idx]+'/*.jpg')
    #         class_dict['size'] = len(class_dict['files'])
    #         class_dcit_lists.append(class_dict)
    #     sorted_list = sorted(class_dcit_lists, key= lambda class_dcit_lists: class_dcit_lists['size'])
    #     return sorted_list


    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, -1]
        label = self.df.iloc[idx, 2]
        image = Image.open(img_path)
        if self.augment_phase and label in self.classes_to_augment:
            image_tensor = self.transform(image)
            #if label in self.classes_to_augment:
            #    image_tensor = self.transform(image)
        else:
            image_tensor = self.transform_NoAug(image)
        # image_tensor = self.transform(image)
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
            
    def __getitem__(self, idx):
        
        img_path = self.df.iloc[idx, -1]
        label = self.df.iloc[idx, 2]
        image = Image.open(img_path)
        
        if label in self.classes_to_augment:
            image_tensor = self.transforms(image)
        else:
            image_tensor = self.transforms.ToTensor()(image)
        
        image_tensor = transforms.ToTensor()(image)
        # x = random.choice(aug_list)
        # image_tensor = augment(image,x)
        
        label_id = torch.tensor(self.class_to_id[str(label)])
        return image_tensor, label_id

import pandas as pd
import torch
import torchvision.transforms as transforms

# Define the threshold
threshold = 0.20

# Load the Excel file and sheet
df = pd.read_excel('combined_report.xlsx', sheet_name='Softmax_Values', index_col=0)

# Function to create the appropriate transforms based on the probabilities
def create_augmentations(df, threshold):
    augmentations_dict = {}
    
    # Iterate over columns (classes)
    for class_name in df.columns:
        transform_list = []
        
        # Iterate over rows (augmentations)
        for augmentation_name, probability in df[class_name].items():
            if probability > threshold:
                if 'VerticalFlip' in augmentation_name:
                    transform_list.append(transforms.RandomVerticalFlip())
                elif 'HorizontalFlip' in augmentation_name:
                    transform_list.append(transforms.RandomHorizontalFlip())
                elif 'GrayScale' in augmentation_name:
                    transform_list.append(transforms.RandomGrayscale())
                elif 'ColorJitter' in augmentation_name:
                    transform_list.append(transforms.ColorJitter())
                elif 'Rotation' in augmentation_name:
                    transform_list.append(transforms.RandomRotation(30))
        
        # Combine the selected transforms
        augmentations_dict[class_name] = transforms.Compose(transform_list)
    
    return augmentations_dict

# Example usage
augmentations = create_augmentations(df, threshold)

# Now you can apply these augmentations to your data
# Example: applying augmentations to an image tensor
image = torch.randn(3, 224, 224)  # Randomly generated image tensor
augmented_image = augmentations['akiec'](image)  # Apply augmentations for class 'akiec'
