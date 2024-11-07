'''
There is a dataset of images of analog clocks. Each image contains a clock showing a different time. 
The goal is to build a model that can predict the time shown in the clock. 
The dataset is available in the data folder. The images are stored in the images.npy file and the corresponding labels are stored in the labels.npy file. 
The labels are in the format (hour, minute). The hour is an integer between 0 and 11 and the minute is an integer between 0 and 59. The task is to build a model that can predict the time shown in the clock from the images. The model should be a Convolutional Neural Network (CNN) and should be trained on the images and labels provided in the dataset. The model should be able to predict the time shown in the clock with an accuracy of at least 90%. The model should be implemented using the TensorFlow library.

'''
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load data
images = np.load('./data/data_big/images.npy')
labels = np.load('./data/data_big/labels.npy')

# Preprocess labels to 720 categories
def label_to_category(label):
    hour, minute = label
    return hour * 60 + minute

categories = np.array([label_to_category(label) for label in labels])
categories = to_categorical(categories, num_classes=720)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, categories, test_size=0.2, random_state=42)

# Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

print(X_train.shape)
# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(720, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Data augmentation
# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# datagen.fit(X_train)

# # Train the model with data augmentation
# history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
#                     steps_per_epoch=len(X_train) / 32, epochs=50,
#                     validation_data=(X_test, y_test))



history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Plot the training process
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()