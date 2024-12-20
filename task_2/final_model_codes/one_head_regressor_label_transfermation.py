import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.initializers import HeNormal, Zeros
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Hyperparameters
batch_size = 128
epochs = 500
learning_rate = 0.0001

# Initializers
kernel_init = HeNormal()
bias_init = Zeros()

# Load data
images = np.load('./task_2/data/data_big/images.npy')
labels = np.load('./task_2/data/data_big/labels.npy')

# Normalize image data
images = images / 255.0

# Image shape
if len(images.shape) == 3:
    images = np.expand_dims(images, axis=-1)

# Convert labels to circular form
def convert_labels_to_circular(labels):
    hours = labels[:, 0] * (2 * np.pi / 12)  # Map hour labels to [0, 2pi]
    minutes = labels[:, 1] * (2 * np.pi / 60)  # Map minute labels to [0, 2pi]
    return np.stack([np.cos(hours), np.sin(hours), np.cos(minutes), np.sin(minutes)], axis=-1)

circular_labels = convert_labels_to_circular(labels)

# Dataset split
X_train, X_temp, y_train, y_temp = train_test_split(images, circular_labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Custom model
def build_custom_model(input_shape):
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(4, activation='linear')(x)  # Output 4 continuous values
    model = Model(inputs=inputs, outputs=outputs)
    return model

input_shape = images[0].shape
model = build_custom_model(input_shape=input_shape)

# Custom loss function, calculating Euclidean distance
def circular_loss(y_true, y_pred):
    hour_cos_sin_true = y_true[:, :2]
    minute_cos_sin_true = y_true[:, 2:]
    hour_cos_sin_pred = y_pred[:, :2]
    minute_cos_sin_pred = y_pred[:, 2:]

    hour_loss = tf.reduce_mean(tf.square(hour_cos_sin_true - hour_cos_sin_pred))
    minute_loss = tf.reduce_mean(tf.square(minute_cos_sin_true - minute_cos_sin_pred))
    return hour_loss + minute_loss

# Compile the model, adding metrics=['mae']
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=circular_loss,
    metrics=['mae']  # Add MAE as an evaluation metric
)

# Print model structure
model.summary()

# Define data augmentation function
def augment(images, labels):
    images = tf.image.random_brightness(images, 0.2)
    images = tf.image.random_contrast(images, 1, 2.0)
    images = tf.image.rot90(images, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    return images, labels

# Build tf.data.Dataset dataset and apply data augmentation
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).map(augment).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Define callback functions
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, mode='min')
checkpoint = ModelCheckpoint('best_Regression_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# Train the model
history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[early_stopping, checkpoint])

# Set path to save weights
weights_path = os.path.join(os.getcwd(), 'Regression.weights.h5')
# Save model weights
model.save_weights(weights_path)

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")

# Prediction example
sample_test_image = X_test[0:1]
predicted_output = model.predict(sample_test_image)

# Convert predicted values back to hours and minutes
def convert_prediction_to_time(pred):
    pred_hour_angle = np.arctan2(pred[1], pred[0])
    pred_minute_angle = np.arctan2(pred[3], pred[2])

    # Process cyclic angles
    pred_hour = (pred_hour_angle * 12 / (2 * np.pi)) % 12
    pred_minute = (pred_minute_angle * 60 / (2 * np.pi)) % 60

    return int(pred_hour), int(pred_minute)

predicted_hour, predicted_minute = convert_prediction_to_time(predicted_output[0])
print(f'Predicted Time: {predicted_hour:02}:{predicted_minute:02}')

# Display original image and prediction result
plt.figure(figsize=(4, 4))
plt.imshow(sample_test_image.squeeze(), cmap='gray')
plt.title(f'Predicted Time: {predicted_hour:02}:{predicted_minute:02}')
plt.axis('off')
plt.show()

# **Plot training curves**
# Plot training and validation loss curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation mean absolute error curves
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('MAE Curve')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.show()
