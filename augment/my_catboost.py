import cv2
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.datasets import make_classification

# Load the dataset from the CSV file
# data = pd.read_csv("../csv/minority_train.csv")


# # Initialize variables to store sampled data as lists
# sampled_image_paths = []
# sampled_class_labels = []

# # Specify the number of samples per class and the total number of samples
# samples_per_class = 20  # Adjust as needed
# total_samples = 100

# # Iterate through unique classes
# unique_classes = data["dx"].unique()
# for class_label in unique_classes:
#     # Select samples for the current class
#     class_data = data[data["dx"] == class_label].head(samples_per_class)
    
#     # Append the sampled image paths and class labels to the result lists
#     sampled_image_paths.extend(class_data["image_pth"].tolist())
#     sampled_class_labels.extend(class_data["dx"].tolist())

# # Initialize an empty list to store images
# images = []

# # Load images using OpenCV and convert to grayscale
# for image_path in sampled_image_paths:
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     images.append(image)

# X = np.array(images)  # Convert to numpy array

# # Convert class labels to numeric values
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(sampled_class_labels)

# # Flatten the image data
# X = X.reshape(X.shape[0], -1)

X, y = make_classification(n_samples=1000, n_features=20, n_classes=5, n_informative=5, random_state=42)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CatBoostClassifier
catboost_classifier = CatBoostClassifier(iterations=500,
                                         depth=6,
                                         learning_rate=0.1,
                                         loss_function='MultiClass',
                                         verbose=0
                                         )

# Train the CatBoostClassifier
catboost_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = catboost_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate and print a classification report
# class_names = label_encoder.classes_
classification_rep = classification_report(y_test, y_pred, target_names=[f'Class_{i}' for i in range(5)])
print(classification_rep)
