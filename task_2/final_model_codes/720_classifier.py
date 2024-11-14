import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load data
images = np.load('./task_2/data/data_big/images.npy')
labels = np.load('./task_2/data/data_big/labels.npy')

# Preprocess labels to 720 categories
def label_to_category(label):
    hour, minute = label
    return hour * 60 + minute

categories = np.array([label_to_category(label) for label in labels])
categories = to_categorical(categories, num_classes=720)

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(images, categories, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalize images
X_train, X_val, X_test = X_train / 255.0, X_val / 255.0, X_test / 255.0

# Reshape images to have a single channel
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Custom metric to check if prediction is within 10 minutes
def within_10_minutes_accuracy(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    
    # Convert labels to hours and minutes
    true_hours = y_true // 60
    true_minutes = y_true % 60
    pred_hours = y_pred // 60
    pred_minutes = y_pred % 60
    
    # Calculate time difference in minutes
    hour_diff = tf.abs(true_hours - pred_hours) * 60
    minute_diff = tf.abs(true_minutes - pred_minutes)
    time_diff = hour_diff + minute_diff
    
    # Adjust for circular behavior (e.g., 11:50 and 12:00 are within 10 minutes)
    time_diff = tf.minimum(time_diff, 720 - time_diff)
    
    # Count as correct if time difference is within 10 minutes
    correct_predictions = tf.reduce_sum(tf.cast(time_diff <= 10, tf.float32))
    accuracy = correct_predictions / tf.cast(tf.size(y_true), tf.float32)
    
    return accuracy

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
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[within_10_minutes_accuracy])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy (within 10 min): {test_accuracy:.4f}")

# Save the model
model.save('./one_head_720_time_classifier.h5')

# Plot training and validation accuracy
# plt.figure(figsize=(12, 5))
plt.title('Model Accuracy (Within 10 Minutes)')
plt.plot(history.history['within_10_minutes_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_within_10_minutes_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
# plt.show()
plt.savefig('one_head_720_time_classifier.png')
