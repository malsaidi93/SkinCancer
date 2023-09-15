import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from imgaug import augmenters as iaa
import warnings 
warnings.filterwarnings("ignore")

# Generate a synthetic multiclass classification dataset (you should replace this with your real data)
X, y = make_classification(n_samples=1000, n_features=20, n_classes=5, n_informative=5, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Convert input data to uint8
X_train = (X_train * 255).astype(np.uint8)
X_test = (X_test * 255).astype(np.uint8)
# Define a list of augmentation techniques
augmentations = [
    iaa.Fliplr(0.5),  # Horizontal flip with 50% probability
    iaa.Flipud(0.5),  # Vertical flip with 50% probability
    iaa.Affine(rotate=(-45, 45)),  # Rotation between -45 and 45 degrees
    iaa.Multiply((0.5, 1.5)),  # Brightness multiplication between 0.5 and 1.5
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
    augmentation_reports[augmentation] = {}
    
    for class_label in np.unique(y_train):
        class_mask = (y_train == class_label)
        augmented_X_train = X_train[class_mask].copy()
        
        # Apply augmentation to the samples of the current class
        augmented_X_train = augmentation.augment_images(augmented_X_train)
        
        # Fit AdaBoost classifier for the current class
        adaboost_classifier.fit(augmented_X_train, y_train[class_mask])
        
        # Evaluate the classifier on the test set
        y_pred = adaboost_classifier.predict(X_test)
        classification_rep = classification_report(y_test, y_pred, target_names=[f'Class_{i}' for i in range(5)], output_dict=True)
        
        # Store the classification report in the dictionary with keys based on augmentation and class
        augmentation_reports[augmentation][f'Class_{class_label}'] = classification_rep

# Print the classification report for each augmentation technique and class
for augmentation, class_reports in augmentation_reports.items():
    print("=" * 40)
    print(f" Augmentation: == {augmentation.name} ==")
    print("=" * 40)
    for class_label, report in class_reports.items():
        print(f"Class {class_label} Classification Report:")
        for metric, value in report.items():
            if metric != 'accuracy':
                print(f"{metric}: {value:.2f}" if isinstance(value, (float, np.float32)) else f"{metric}: {value}")
        print("=" * 40)
