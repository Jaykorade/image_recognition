import pandas as pd
import requests
import os
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Load dataset (CSV/Excel with URL and Label columns)
df = pd.read_excel("pexels_images_combined.xlsx")  # Change to .xlsx if needed
df.columns = ["image_url", "label"]

# Create directories to store images
os.makedirs("images/cat", exist_ok=True)
os.makedirs("images/dog", exist_ok=True)

# Download images
for index, row in df.iterrows():
    response = requests.get(row["image_url"])
    if response.status_code == 200:
        filename = f'images/{row["label"]}/{index}.jpg'
        with open(filename, "wb") as f:
            f.write(response.content)

print("All images downloaded!")
# Image size and batch size
IMG_SIZE = (128, 128)

# Load images and labels
def load_data(folder):
    images = []
    labels = []
    for label in ["cat", "dog"]:
        path = os.path.join(folder, label)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, IMG_SIZE)
            images.append(img)
            labels.append(0 if label == "cat" else 1)  # 0=Cat, 1=Dog
    return np.array(images), np.array(labels)

# Load dataset
X, y = load_data("images")
X = X / 255.0  # Normalize images

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)