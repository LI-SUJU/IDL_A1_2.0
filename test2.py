import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt

# Load data
images = np.load('./data/data_big/images.npy')
labels = np.load('./data/data_big/labels.npy')

# Preprocess labels to 720 categories
def label_to_category(label):
    hour, minute = label
    return hour * 60 + minute

categories = np.array([label_to_category(label) for label in labels])
categories = tf.keras.utils.to_categorical(categories, num_classes=720)

# Split data into train, val, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(images, categories, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalize and reshape images
X_train, X_val, X_test = X_train / 255.0, X_val / 255.0, X_test / 255.0
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
)

# Model using transfer learning
base_model = MobileNetV2(input_shape=(X_train.shape[1], X_train.shape[2], 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model

model = Sequential([
    tf.keras.layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x)),
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(720, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=30,
    validation_data=(X_val, y_val)
)

# Save the model
model.save('./improved_720_head_model.h5')

# Plot training and validation accuracy
# plt.figure(figsize=(12, 5))
plt.title('Model Accuracy')
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
# save the plot
plt.savefig('./improved_720_head_model_accuracy_plot.png')