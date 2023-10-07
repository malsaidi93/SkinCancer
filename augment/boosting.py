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
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import pandas as pd
import imgaug as ia
import cv2
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")

# Load the dataset from the CSV file
data = pd.read_csv("../csv/minority_train.csv")

# Initialize variables to store sampled data as lists
sampled_image_paths = []
sampled_class_labels = []

# Specify the number of samples per class and the total number of samples
samples_per_class = 10  # Adjust as needed

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

print('splitting data')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = (X_train * 255).astype(np.uint8)
X_test = (X_test * 255).astype(np.uint8)
base_classifier = DecisionTreeClassifier(max_depth=2)
adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators=50, random_state=42)
adaboost_classifier.fit(X_train, y_train)
y_pred = adaboost_classifier.predict(X_test)
classification_rep = classification_report(y_test, y_pred, target_names=[i for i in range(0,len(np.unique(y_train)))], output_dict=True)

proba = adaboost_classifier.predict_proba(X_test)

print('='*50)
print(tabulate(proba))