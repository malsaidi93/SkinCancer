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
import warnings
warnings.filterwarnings("ignore")


# Load the dataset from the CSV file
data = pd.read_csv("../csv/minority_train.csv")

# Initialize variables to store sampled data as lists
sampled_image_paths = []
sampled_class_labels = []

# Specify the number of samples per class and the total number of samples
samples_per_class = 100  # Adjust as needed
total_samples = 500

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

# Load images using OpenCV and convert to grayscale
for image_path in sampled_image_paths:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    images.append(image)


X = np.array(images)  # Convert to numpy array

# Convert class labels to numeric values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(sampled_class_labels)

# Flatten the image data
X = X.reshape(X.shape[0], -1)

# Split the dataset into training and testing sets
print('splitting data')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Convert input data to uint8
X_train = (X_train * 255).astype(np.uint8)
X_test = (X_test * 255).astype(np.uint8)
# Define a list of augmentation techniques
augmentations = [
        None,
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Multiply(0.5),
        iaa.GaussianBlur((0, 3.0)),
        iaa.CropAndPad(percent=(-0.05, 0.1), pad_mode=ia.ALL, pad_cval=(0, 255)),
        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
        iaa.Dropout((0.01, 0.1), per_channel=0.5),
        iaa.Invert(0.05, per_channel=True),
        iaa.AddToHueAndSaturation((-20, 20)),
        iaa.Grayscale(alpha=(0.0, 1.0)),
        ]

# Define the base classifier (Decision Tree)
base_classifier = DecisionTreeClassifier(max_depth=1)

# Initialize AdaBoostClassifier with the base classifier
adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators=50, random_state=42)

# Create a dictionary to store classification reports for each augmentation and class
augmentation_reports = {}

# Train AdaBoost with each augmentation technique for each class and record classification report
num_iterations = 10  # Number of iterations to test each augmentation
for augmentation in augmentations:
    print(f'AUGMENTATION: {augmentation.name}')
    if augmentation is None:
        aug = 'None'
        augmented_X_train = X_train
    else:
        aug = augmentation.name
        augmented_X_train = augmentation.augment_images(X_train)
    adaboost_classifier.fit(augmented_X_train, y_train)
    y_pred = adaboost_classifier.predict(X_test)
    classification_rep = classification_report(y_test, y_pred, target_names=[i for i in range(0,len(np.unique(y_train)))], output_dict=True)
    cf = confusion_matrix(y_test, y_pred)
    print(f'confusion matrix: \n{cf}')
    # Store the classification report in the dictionary with keys based on augmentation and class
    augmentation_reports[augmentation] = classification_rep

with open(f'./reports/metrics_{aug}.txt', 'w+')as metrics:
    metrics.write(str(augmentation_reports))
# Print the classification report for each augmentation technique and class
for augmentation, class_reports in augmentation_reports.items():
    print("=" * 40)
    print(f" Augmentation: == {augmentation.name} ==")
    print("=" * 40)
    for metric, value in class_reports.items():
        print(f"{metric}: {value:.2f}" if isinstance(value, (float, np.float32)) else f"{metric}: {value}")
        print("=" * 40)


