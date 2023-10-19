import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from imgaug import augmenters as iaa
import imgaug as ia
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import pandas as pd
import cv2
import warnings
import json
import random
import os
import time
import itertools
import pickle
import matplotlib.pyplot as plt
import plotext as plx
from uniplot import histogram
warnings.filterwarnings("ignore")

def plot_confusion_matrix(cm, class_names, filename):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=8, horizontalalignment='right')
    plt.yticks(tick_marks, class_names, fontsize=8)
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color, fontsize=7)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'./reports_all/{filename}')
    # return figure

# Load the dataset from the CSV file
data = pd.read_csv("../csv/minority_train.csv")

# Initialize variables to store sampled data as lists
sampled_image_paths = []
sampled_class_labels = []

# Specify the number of samples per class and the total number of samples
samples_per_class = 50  # Adjust as needed

# Iterate through unique classes
unique_classes = data["dx"].unique()
for class_label in unique_classes:
    # Select samples for the current class
    class_data = data[data["dx"] == class_label].head(samples_per_class)
    
    # Append the sampled image paths and class labels to the result lists
    sampled_image_paths.extend(class_data["image_pth"].tolist())
    sampled_class_labels.extend(class_data["dx"].tolist())


# Initialize an empty list to store images
images = []
for image_path in sampled_image_paths:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    images.append(image)

X = np.array(images)  # Convert to numpy array
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(sampled_class_labels)
X = X.reshape(X.shape[0], -1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = (X_train * 255).astype(np.uint8)
X_test = (X_test * 255).astype(np.uint8)

distribution = np.bincount(y_train)
print(f'Original Distribution : {distribution}')

base_classifier = DecisionTreeClassifier(max_depth=2)
adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators=50, random_state=42)

# c = {
#     0: {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0},
#     1: {'precision': 0.0, 'recall': 1.0, 'f1-score': 0.0, 'support': 0.0},
#     2: {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0},
#     3: {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0},
#     4: {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0},
#     5: {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0},
#     'accuracy': 0.0,
#     'macro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0},
#     'weighted avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0}
#     }

augmentations = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.GaussianBlur((0, 3.0)),
        iaa.CropAndPad(percent=(-0.05, 0.1), pad_mode=ia.ALL, pad_cval=(0, 255)),
        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))
])

# Your f1_scores and exclude dictionary
f1_scores = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
exclude = ['accuracy', 'macro avg', 'weighted avg']  # Corrected 'accurcy' to 'accuracy'

with open('./reports_all/metrics.txt', 'a+')as metrics:
            metrics.write(str(f1_scores))
            metrics.write('\n')
# Loop through the f1_scores dictionary
print('=' * 50)
counter = 0
min_f1_score = 0.3
flag = True
load = False
while flag:
    flag = False
    for key, value in f1_scores.items():
        if value <= min_f1_score:
            clf = sorted(os.listdir('./classifier/'))
            if len(clf) > 0:
                with open(f'./classifier/{clf[-1]}','rb') as f:
                    print(f'Loading {clf[-1]} classifier...')
                    print(f'Estimator Weight: {adaboost_classifier.estimator_weights_}')
                    adaboost_classifier = pickle.load(f)
            
            start = time.time()
            adaboost_classifier.fit(X_train, y_train)
            end = time.time()
            # save
            with open(f'./classifier/model_{counter}.pkl','wb') as f:
                print('Saving model...')
                pickle.dump(adaboost_classifier,f)
                counter += 1
            y_pred = adaboost_classifier.predict(X_test)
            classification_rep = classification_report(
                y_test, y_pred, target_names=[str(i) for i in range(0, len(np.unique(y_train)))], output_dict=True
            )
            print('-' * 50)
            print('Classification Report:')
            print('-' * 50)
            for key, value in classification_rep.items():
                print(f"{key} ==> {value}")
            print('-' * 50)
            
            cf = confusion_matrix(y_test, y_pred)
            plot_confusion_matrix(cf, [i for i in range(0,len(np.unique(y_train)))], f'model_{counter-1}.png')
            
            # Update the f1_scores dictionary
            for each_class in classification_rep.keys():
                if each_class not in exclude and each_class.isdigit():
                    if int(each_class) in f1_scores.keys():
                        f1_scores[int(each_class)] = classification_rep[each_class]['f1-score']

            # Augment Data for classes with f1-score less than 0.3
            for each_class in f1_scores.keys():
                if f1_scores[each_class] < min_f1_score:
                    # print(f'Class: {each_class} F1-score: {classification_rep[str(each_class)]["f1-score"]}')
                    
                    selected_class = (y_train == each_class)
                    
                    images_copy = X_train[selected_class].copy()
                    # combine_selected_image = np.array([])
                    # for idx in range(0, 10):
                    #     random_image  = random.randint(0, len(images_copy))
                    #     selected_image = images_copy[random_image]
                    #     np.concatenate((combine_selected_image, selected_image))
                        
                        
                    augmented_images = augmentations(images = images_copy)  # Assuming you have a function augmentations.augment_images
                    X_train = np.concatenate((X_train, augmented_images[:10]), axis=0)
                    y_train = np.concatenate((y_train, np.full((len(augmented_images[:10]),), each_class)), axis=0)
                    flag = True
            distribution = np.bincount(y_train)
            print(f'Distribution : {distribution}')
            print('=' * 50)
            
            with open('./reports_all/metrics.txt', 'a+')as metrics:
                metrics.write(str(classification_rep))
                metrics.write('\n')
                metrics.write(str(distribution))
                metrics.write('\n')
            
            print(f'Total Time: {end - start}')
            load = True
            
# proba = adaboost_classifier.predict_proba(X_test)

# print('='*50)
# print(tabulate(proba))