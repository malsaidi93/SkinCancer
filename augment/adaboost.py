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
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import pandas as pd
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(f"ytest: {np.unique(y_test)}")
print(f"ytrain: {np.unique(y_train)}")

class_distribution_test = np.bincount(y_test)
class_distribution_train = np.bincount(y_train)
print("Class distribution in y_test:", class_distribution_test)
print("Class distribution in y_train:", class_distribution_train)

# Generate a synthetic multiclass classification dataset (you should replace this with your real data)
# X, y = make_classification(n_samples=1000, n_features=20, n_classes=5, n_informative=5, random_state=42)


# Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = (X_train * 255).astype(np.uint8)
X_test = (X_test * 255).astype(np.uint8)

# Define a list of augmentation techniques
augmentations = [
    iaa.Fliplr(0.5),  # Horizontal flip with 50% probability
    iaa.Flipud(0.5),  # Vertical flip with 50% probability
    # iaa.Affine(rotate=(-45, 45)),  # Rotation between -45 and 45 degrees
    iaa.Multiply((0.5, 1.5)),  # Brightness multiplication between 0.5 and 1.5
]

# Define the base classifier (Decision Tree)
base_classifier = DecisionTreeClassifier(max_depth=1)

# Initialize AdaBoostClassifier with the base classifier

adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators=50, random_state=42)
# adaboost_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
# adaboost_classifier = CatBoostClassifier(verbose=0, n_estimators=100)
# adaboost_classifier = xgb.XGBClassifier(n_estimators=100,
                                        # max_depth=6,
                                        # learning_rate=0.1,
                                        # verbosity=0,
                                        # objective='multi:softmax',
                                        # num_class=6,
                                        # random_state=42)

# Create a dictionary to store classification reports for each augmentation and class
augmentation_reports = {}

unique_classes_test = np.unique(y_test)


# Train AdaBoost with each augmentation technique for each class and record classification report
num_iterations = 10  # Number of iterations to test each augmentation
for augmentation in augmentations:
    augmentation_reports[augmentation] = {}
    print(f"Augmentation: {augmentation.name}")
    for class_label in np.unique(y_train):
        print(f"Class: {class_label}")
        class_mask = (y_train == class_label)
        augmented_X_train = X_train[class_mask].copy()
        
        # Apply augmentation to the samples of the current class
        augmented_X_train = augmentation.augment_images(augmented_X_train)
        
        # Fit AdaBoost classifier for the current class
        adaboost_classifier.fit(augmented_X_train, y_train[class_mask])
        
        # Evaluate the classifier on the test set
        y_pred = adaboost_classifier.predict(X_test)
        # Check unique classes in y_test and y_pred
        print(f'Unique classes test : {np.unique(y_test)}')
        print(f'Unique classes pred : {np.unique(y_pred)}')
        
        # # Confirm that both sets of unique classes match
        # if np.array_equal(unique_classes_test, unique_classes_pred):
        #     target_names = [f'Class_{i}' for i in unique_classes_test]
        # else:
        #     print("Mismatch in unique classes between y_test and y_pred.")
        
        # Then, when calling classification_report, pass the target_names parameter
        classification_rep = classification_report(y_test, y_pred, target_names=unique_classes_test, output_dict=True)
        # Store the classification report in the dictionary with keys based on augmentation and class
        augmentation_reports[augmentation][f'Class_{class_label}'] = classification_rep

# Print the classification report for each augmentation technique and class
for augmentation, class_reports in augmentation_reports.items():
    print("#" * 40)
    print(f"\nAugmentation:  == {augmentation.name} ==\n")
    print("=" * 40)
    for class_label, report in class_reports.items():
        print(f"Class {class_label} Classification Report:")
        for metric, value in report.items():
            if metric != 'accuracy' and f"Class_{metric}" == class_label:
                print(f"metric: Class_{metric}, class_label: {class_label}")
                print(f"{metric}: {value:.2f}" if isinstance(value, (float, np.float32)) else f"{metric}: {value}")
        print("=" * 40)
    print("#" * 40)


# print("*" * 40)
# print(class_reports)
# print("*" * 40)

