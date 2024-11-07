import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

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

# Reshape the images to have a single channel
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# # 定义数据增强函数
# def augment(images, labels):
#     images = tf.image.random_brightness(images, 0.2)  # 调整亮度
#     images = tf.image.random_contrast(images, 1, 2.0)  # 增强对比度
#     images = tf.image.rot90(images, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))  # 随机旋转
#     return images, labels  # 返回增强后的图像和原始标签

# def augment4valSet(images, labels):
#     images = tf.image.random_brightness(images, 0.2)  # 调整亮度
#     images = tf.image.random_contrast(images, 1, 2.0)  # 增强对比度
#     return images, labels  # 返回增强后的图像和原始标签

# # Apply augmentation
# X_train, y_train = augment(X_train, y_train)
# X_test, y_test = augment4valSet(X_test, y_test)

def circular_loss(y_true, y_pred):
    # Convert one-hot encoded labels back to integers
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    
    # Convert labels to hours and minutes
    true_hours = tf.cast(y_true // 60, tf.float32)
    true_minutes = tf.cast(y_true % 60, tf.float32)
    pred_hours = tf.cast(y_pred // 60, tf.float32)
    pred_minutes = tf.cast(y_pred % 60, tf.float32)
    
    # Calculate the difference in minutes
    diff_hours = tf.abs(true_hours - pred_hours)
    diff_minutes = tf.abs(true_minutes - pred_minutes)
    diff = diff_hours * 60 + diff_minutes
    
    # Adjust for circular difference
    diff = tf.minimum(diff, 720 - diff)
    
    # Calculate L2 norm
    l2_norm = tf.square(diff)
    
    # Calculate the final loss
    loss = tf.reduce_mean(l2_norm)
    
    return loss

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])),
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
model.compile(optimizer='adam', loss=circular_loss, metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Plot the training process
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
