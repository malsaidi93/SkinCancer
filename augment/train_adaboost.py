import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
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
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def train(csv, augmentation=None, total_samples=600, samples_per_class=100):
    
    # Load the dataset from the CSV file
    data = pd.read_csv(csv)

    sampled_image_paths = []
    sampled_class_labels = []
    # samples_per_class = 100  # Adjust as needed
    # total_samples = 600

    unique_classes = data["dx"].unique()
    for class_label in unique_classes:
        class_data = data[data["dx"] == class_label].head(samples_per_class)
        sampled_image_paths.extend(class_data["image_pth"].tolist())
        sampled_class_labels.extend(class_data["dx"].tolist())
    images = []

    for image_path in sampled_image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        images.append(image)

    X = np.array(images)  # Convert to numpy array
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(sampled_class_labels)

    X = X.reshape(X.shape[0], -1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"ytest: {np.unique(y_test)}")
    print(f"ytrain: {np.unique(y_train)}")

    class_distribution_test = np.bincount(y_test)
    class_distribution_train = np.bincount(y_train)
    print("Class distribution in y_test:", class_distribution_test)
    print("Class distribution in y_train:", class_distribution_train)

    X_train = (X_train * 255).astype(np.uint8)
    X_test = (X_test * 255).astype(np.uint8)

    base_classifier = DecisionTreeClassifier(max_depth=1)
    adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators=50, random_state=42)
    augmentation_reports = {}
    unique_classes_test = np.unique(y_test)
    num_iterations = 10  # Number of iterations to test each augmentation
    consolidated_x = []
    consolidated_y = []

    for class_label in np.unique(y_train):
        print('=' * 40)
        print(f"Class: {class_label}")
        class_mask = (y_train == class_label)
        other_mask = (y_train != class_label)
        augmented_X_train = X_train[class_mask].copy()
        non_augmented_X_train = X_train[other_mask].copy()
        augmented_X_train = augmentation.augment_images(augmented_X_train)
        all_data = np.concatenate((augmented_X_train, non_augmented_X_train), axis=0)
        all_labels = np.concatenate((y_train[class_mask], y_train[other_mask]), axis=0)
        adaboost_classifier.fit(all_data, all_labels)
        y_pred = adaboost_classifier.predict(X_test)
        # print(f'Classes test : {y_test}')
        # print(f'Classes pred : {y_pred}')
        classification_rep = classification_report(y_test, y_pred, target_names=unique_classes_test, output_dict=True)
        # print(f'Classification report: {classification_rep}')
        for key, value in classification_rep.items():
            print(f'{key}: {value}') 
        print('=' * 40)
        
        augmentation_reports[f'Class_{class_label}'] = classification_rep

    # Print the classification report for each augmentation technique and class
    # for class_report, metric in augmentation_reports.items():
    #     print("#" * 40)
    #     print(f"\Class:  == {class_report} ==\n")
    #     print("=" * 40)
    #     if metric != 'accuracy':
    #         # print(f"metric: {metric}, value: {value}")
    #         print(f"{class_report}: {metric:.2f}" if isinstance(metric, (float, np.float32)) else f"{class_report}: {metric}")
        
    with open('./reports/metrics_Multiply_100Images.txt', 'w+')as metrics:
        metrics.write(str(augmentation_reports))


if __name__ == "__main__":
    csv = "../csv/minority_train.csv"
    augmentations = [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Multiply(0.5),
        iaa.GaussianBlur(0, 2.0),
        iaa.CropAndPad(percent=(-0.05, 0.1), pad_mode=ia.ALL, pad_cval=(0, 255)),
        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
        iaa.Dropout((0.01, 0.1), per_channel=0.5),
        iaa.Invert(0.05, per_channel=True),
        iaa.AddToHueAndSaturation((-20, 20)),
        iaa.Grayscale(alpha=(0.0, 1.0)),
        ]
    total_samples = 600
    sample_per_class = 100
    train(csv, augmentations[2], total_samples, sample_per_class)