import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Paths to the dataset folder and CSV file
data_dir = r"C:\Users\HP\Desktop\ML\train\train"  # Path to the 'train' folder
csv_file = r"C:\Users\HP\Desktop\ML\Training_set_covid.csv"  # Path to CSV file

# Image size for resizing
image_size = (150, 150)

# Load CSV file to get image names and labels
df = pd.read_csv(csv_file)

# Function to load images and labels from the directory
def load_images_labels_from_csv(data_dir, csv_file, image_size):
    images = []
    labels = []
    
    for index, row in df.iterrows():
        file_name = row['filename']  # Update this to the correct column name ('filename')
        label = row['label']  # Update this to the correct column name ('label')
        
        img_path = os.path.join(data_dir, file_name)
        
        if os.path.exists(img_path):  # Check if the image file exists
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Load the images and labels from the CSV and directory
X, y = load_images_labels_from_csv(data_dir, csv_file, image_size)

# Normalize the images
X = X / 255.0  # Normalize pixel values to [0,1]
X = X.reshape(-1, image_size[0], image_size[1], 1)  # Add channel dimension (grayscale)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation to enhance the robustness of the model
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=20,
                    callbacks=[early_stop])

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# Plot the training history (accuracy and loss)
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

model.save(r"C:\Users\HP\Desktop\ML\covid_model.h5")