import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Load the dataset
images = np.load('data/images.npy')
labels = np.load('data/labels.npy')

# Normalize the images
images = images / 255.0

# Split the labels into hours and minutes
hours = labels[:, 0]
minutes = labels[:, 1]

# Define the input shape
input_shape = images.shape[1:]

# Define the model
input_layer = Input(shape=input_shape)

# Convolutional layers
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# Hour prediction head
hour_output = Dense(12, activation='softmax', name='hour_output')(x)

# Minute prediction head
minute_output = Dense(60, activation='softmax', name='minute_output')(x)

# Define the model with two outputs
model = Model(inputs=input_layer, outputs=[hour_output, minute_output])

# Custom loss functions
def circular_loss_hour(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.reduce_sum(y_pred * tf.range(12, dtype=tf.float32), axis=-1)
    diff = tf.abs(y_true - y_pred)
    return tf.reduce_mean(tf.minimum(diff, 12 - diff))

def circular_loss_minute(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.reduce_sum(y_pred * tf.range(60, dtype=tf.float32), axis=-1)
    diff = tf.abs(y_true - y_pred)
    return tf.reduce_mean(tf.minimum(diff, 60 - diff))

# Compile the model
model.compile(optimizer='adam',
              loss={'hour_output': circular_loss_hour, 'minute_output': circular_loss_minute},
              metrics={'hour_output': 'accuracy', 'minute_output': 'accuracy'})

# Train the model
history = model.fit(images, {'hour_output': hours, 'minute_output': minutes}, epochs=20, batch_size=32, validation_split=0.2)

# Save the model
model.save('clock_time_prediction_model_multihead.h5')

# Plot the training process
def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['hour_output_accuracy'])
    plt.plot(history.history['val_hour_output_accuracy'])
    plt.plot(history.history['minute_output_accuracy'])
    plt.plot(history.history['val_minute_output_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train Hour', 'Val Hour', 'Train Minute', 'Val Minute'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['hour_output_loss'])
    plt.plot(history.history['val_hour_output_loss'])
    plt.plot(history.history['minute_output_loss'])
    plt.plot(history.history['val_minute_output_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Hour', 'Val Hour', 'Train Minute', 'Val Minute'], loc='upper left')

    plt.show()

# Call the function to plot the training history
plot_training_history(history)