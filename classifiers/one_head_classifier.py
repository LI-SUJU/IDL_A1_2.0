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

# Alternative circular metric instead of using in loss
def circular_metric(y_true, y_pred):
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
    
    # Calculate the final mean difference
    metric = tf.reduce_mean(l2_norm)
    
    return metric

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

import tensorflow as tf

# Custom "common sense" time difference loss function with 10-minute threshold
import tensorflow as tf
from tensorflow.keras import backend as K

def circular_time_loss(y_true, y_pred):
    # Convert y_true to class indices (from one-hot encoding)
    y_true = tf.argmax(y_true, axis=-1)  # Shape (batch_size,)
    y_true = tf.cast(y_true, tf.float32)

    # Get predicted classes (use argmax to get the predicted class indices)
    pred_classes = tf.argmax(y_pred, axis=-1)  # Shape (batch_size,)
    pred_classes = tf.cast(pred_classes, tf.float32)

    # Split into hours and minutes
    true_hours = tf.math.floordiv(y_true, 60)  # Integer division to extract hours
    true_minutes = y_true % 60  # Extract minutes
    pred_hours = tf.math.floordiv(pred_classes, 60)  # Integer division to extract hours
    pred_minutes = pred_classes % 60  # Extract minutes
    
    # Circular difference for hours (12-hour wrap)
    hour_diff = tf.abs(true_hours - pred_hours)
    hour_diff = tf.minimum(hour_diff, 12 - hour_diff)  # Adjust for circular behavior

    # Circular difference for minutes (60-minute wrap)
    minute_diff = tf.abs(true_minutes - pred_minutes)
    minute_diff = tf.minimum(minute_diff, 60 - minute_diff)  # Adjust for circular behavior
    
    # Combine hours and minutes into total circular difference
    total_diff = hour_diff * 60 + minute_diff  # Total difference in minutes

    # Apply threshold (e.g., 10 minutes) where loss is minimized
    zero_loss_mask = tf.cast(total_diff <= 10, tf.float32)  # Loss should be zero if within 10 minutes
    
    # Use sparse categorical crossentropy for base loss
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    # Apply circular weight to the loss
    weighted_loss = loss * (1 - zero_loss_mask / 60)  # Scale the loss by circular difference

    return K.mean(weighted_loss)



# Compile the model with the new loss function
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy', circular_metric])


# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))



# Save the model
model.save('./720_head_model.h5')


# Plot the training process
# plot title
plt.figure(figsize=(12, 5))
plt.title('Model Accuracy')
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
