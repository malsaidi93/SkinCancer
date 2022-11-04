
import os, sys, glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from PIL import Image
import random
# from do_augmentation import augment



class SkinCancer(Dataset):
    """Skin Cancer Dataset."""

    def __init__(self, root_dir, meta, transform=None):
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
        self.image_paths = self.df['image_path'].to_list()
        self.image_ids = self.df['image_id'].to_list()
        self.classes = self.df['dx'].unique().tolist()

        
        self.class_id = {i:j for i, j in enumerate(self.classes)}

        self.class_to_id = {value:key for key,value in self.class_id.items()}
        


        # self.file_names_ids = {i:v for v,i in enumerate(self.file_names)}
        
        
        
        

    def __len__(self):
        return len(self.image_paths)
    

    
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
        
        # aug_list = [1,2,3,4,5,6,7,8]

        img_path = self.df.iloc[idx, -1]
        label = self.df.iloc[idx, 2]

        image = Image.open(img_path)
        
        image = transforms.Resize(size=(224,224))(image)
        image_tensor = transforms.ToTensor()(image)

        # x = random.choice(aug_list)
        # image_tensor = augment(image,x)
        
        label_id = self.class_to_id[str(label)]
        return image_tensor, label_id