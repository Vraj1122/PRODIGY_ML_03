import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set dataset directory
# Update this path to the actual path where the dataset is located
dataset_dir = 'C:/Users/vrajp/PycharmProjects/PRODIGY_ML_03/SVM/Dataset'  # Change this to the path where you have the dataset

# Parameters
img_size = (128, 128)  # Resize images to 64x64 pixels
categories = ['cat', 'dog']

# Load and preprocess the data
def load_data(dataset_dir, img_size, categories):
    data = []
    labels = []
    for category in categories:
        category_path = os.path.join(dataset_dir, category)
        label = categories.index(category)
        for img_name in os.listdir(category_path):
            try:
                img_path = os.path.join(category_path, img_name)
                img = load_img(img_path, target_size=img_size)
                img_array = img_to_array(img)
                img_array = img_array / 255.0  # Normalize pixel values
                data.append(img_array.flatten())  # Flatten the image
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(data), np.array(labels)

# Load data
data, labels = load_data(dataset_dir, img_size, categories)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

# Train the SVM model
svm = SVC(kernel="poly", degree=5, gamma="scale")
svm.fit(X_train, y_train)

# Make predictions
y_pred = svm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=categories))

# Visualize some sample predictions
def visualize_predictions(X_test, y_test, y_pred, categories, n_samples=5):
    plt.figure(figsize=(10, 10))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i + 1)
        plt.imshow(X_test[i].reshape(img_size[0], img_size[1], 3))
        plt.title(f"Actual: {categories[y_test[i]]}\nPredicted: {categories[y_pred[i]]}")
        plt.axis('off')
    plt.show()

# Visualize predictions
visualize_predictions(X_test, y_test, y_pred, categories)
