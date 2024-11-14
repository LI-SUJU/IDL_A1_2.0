import os

from keras.src.layers import Flatten

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import numpy as np
import tensorflow as tf
from tensorflow.image import adjust_contrast
from tensorflow.keras import layers, Input, models, Model, backend
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import HeNormal, Zeros
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True   
# sess = tf.Session(config=config)

# Defining Hyperparameters
batch_size = 128
epochs = 500
learning_rate = 0.001

# Defining Initializers
kernel_init = HeNormal()  # He Normal initialization
bias_init = Zeros()  # The bias is initialized to 0

# Loading data
images = np.load('./task_2/data/data_big/images.npy')
labels = np.load('./task_2/data/data_big/labels.npy')

# images = tf.squeeze(tf.image.resize(tf.expand_dims(images, -1), [299, 299]))

# Normalize image data, scaling pixel values ​​to [0, 1]
images = images / 255.0

# If the image is grayscale, add a dimension at the end to fit Keras's input shape requirements
if len(images.shape) == 3:
    images = np.expand_dims(images, axis=-1)  # images = np.repeat(images, 3, axis=-1)

# def preprocess_image(image):
#     images = tf.image.adjust_brightness(image, -0.2)  # Adjust brightness
#     images = adjust_contrast(images, contrast_factor=2.5)  # Enhance contrast
#     images = adjust_gamma(images, gain=1.0, gamma=4)
#     images = tf.clip_by_value(images, 0.0, 1.0)
#     return images

# images = preprocess_image(images).numpy()

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Separate hours and minutes from labels
y_train_hours = y_train[:, 0]
y_train_minutes = y_train[:, 1]
y_val_hours = y_val[:, 0]
y_val_minutes = y_val[:, 1]
y_test_hours = y_test[:, 0]
y_test_minutes = y_test[:, 1]


# plt.hist(y_train_hours, bins=12)  # Check the distribution of hour labels
# plt.hist(y_train_minutes, bins=60)  # Check the distribution of minute labels
# plt.show()
# exit(0)
def build_custom_model(input_shape):
    # Input Layer
    inputs = Input(shape=input_shape)
    # Convolutional and pooling layers
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    # x = layers.BatchNormalization()(x)  # add Batch Normalization
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    # x = layers.BatchNormalization()(x)  # add Batch Normalization
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    # x = layers.BatchNormalization()(x)  # add Batch Normalization
    # # Global Average Pooling Layer
    # x = layers.GlobalAveragePooling2D()(x)
    # Defining Output
    model = models.Model(inputs=inputs, outputs=x)
    return model


input_shape = images[0].shape
base_model = build_custom_model(input_shape=input_shape)
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=None, pooling='avg')
# base_model = VGG16(weights='imagenet', include_top=False)
# Set path to save weights
weights_path = os.path.join(os.getcwd(), 'DIY.weights.h5')
# Save model weights
base_model.save_weights(weights_path)
# Define optimizer with adjustable learning rate
# adam = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# # Freeze layers
# for layer in base_model.layers:
#     layer.trainable = False

# Add decoder
encoder = base_model.output
decoder_1 = Flatten()(encoder)
decoder_1 = Dense(32, activation='relu')(decoder_1)
decoder_1 = Dropout(0.1)(decoder_1)
decoder_hour = Dense(12, activation='sigmoid', name='hour_output')(decoder_1)
decoder_2 = Flatten()(encoder)
decoder_2 = Dense(256, activation='relu')(decoder_2)
decoder_2 = Dropout(0.1)(decoder_2)
decoder_min = Dense(60, activation='sigmoid', name='minute_output')(decoder_2)

# Construct multi-head model
model = Model(inputs=base_model.input, outputs=[decoder_hour, decoder_min])


# exit(0)


def circular_hour_loss(y_true, y_pred):
    # Convert y_true to integer labels
    y_true = tf.cast(y_true, tf.float32)

    # Find the most likely predicted class
    pred_classes = tf.argmax(y_pred, axis=-1)
    pred_classes = tf.cast(pred_classes, tf.float32)

    # Calculate forward error and backward error (12-hour format)
    forward_diff = tf.abs(pred_classes - y_true)
    backward_diff = 12 - forward_diff

    # Interpolate based on the minimum error, with the smaller error category receiving higher weight
    circular_diff = tf.minimum(forward_diff, backward_diff)

    # Calculate interpolated loss value
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    # Weight the loss with circular error
    weighted_loss = loss * (1 - circular_diff / 12)

    return backend.mean(weighted_loss)


def circular_minute_loss(y_true, y_pred):
    # Convert y_true to integer labels
    y_true = tf.cast(y_true, tf.float32)

    # Find the most likely predicted class
    pred_classes = tf.argmax(y_pred, axis=-1)
    pred_classes = tf.cast(pred_classes, tf.float32)

    # Calculate forward error and backward error (12-hour format)
    forward_diff = tf.abs(pred_classes - y_true)
    backward_diff = 60 - forward_diff

    # Interpolate based on the minimum error, with the smaller error category receiving higher weight
    circular_diff = tf.minimum(forward_diff, backward_diff)

    # If the error is within 5 minutes, consider the loss as 0
    zero_loss_mask = tf.cast(circular_diff <= 10, tf.float32)

    # Calculate interpolated loss value
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    # Weight the loss with circular error
    weighted_loss = loss * (1 - zero_loss_mask / 60) * (1 - zero_loss_mask)

    return backend.mean(weighted_loss)


# Compile model
model.compile(optimizer="rmsprop",
              # loss={'hour_output': 'sparse_categorical_crossentropy', 'minute_output': 'sparse_categorical_crossentropy'},
              loss={'hour_output': circular_hour_loss, 'minute_output': circular_minute_loss},
              metrics={'hour_output': 'accuracy', 'minute_output': 'accuracy'})

# Print model structure
model.summary()


# Define data augmentation function
def augment(images, labels):
    images = tf.image.random_brightness(images, 0.2)  # Adjust brightness
    images = tf.image.random_contrast(images, 1, 2.0)  # Enhance contrast
    images = tf.image.rot90(images, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))  # Random rotation
    return images, labels  # Return augmented images and original labels


# Build tf.data.Dataset dataset and apply data augmentation
train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, {'hour_output': y_train_hours, 'minute_output': y_train_minutes}))
train_dataset = train_dataset.shuffle(buffer_size=1024).map(augment).batch(batch_size).prefetch(
    tf.data.experimental.AUTOTUNE)

# first_element = next(iter(train_dataset.unbatch()))
# import matplotlib.pyplot as plt
# debug_show_pic = first_element[0]
# # This line is to change shape, delete if unnecessary
# plt.imshow(debug_show_pic, cmap='gray')
# plt.show()
# exit(0)


val_dataset = tf.data.Dataset.from_tensor_slices((X_val, {'hour_output': y_val_hours, 'minute_output': y_val_minutes}))
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Define callback functions
early_stopping = EarlyStopping(monitor='val_minute_output_accuracy', patience=20, restore_best_weights=True, mode='max')
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_minute_output_accuracy')

# Train the model and add callback
history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[early_stopping, checkpoint])

model.save('classification_multihead_without_labels_train_model.h5')

# Plotting the total loss curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Total training loss')
plt.plot(history.history['val_loss'], label='Validation total loss')
plt.title('Multi-head classification total loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot the loss curve for each output
plt.subplot(1, 2, 2)
plt.plot(history.history['hour_output_loss'], label='Output training loss in hours')
plt.plot(history.history['val_hour_output_loss'], label='Hourly output validation loss')
plt.plot(history.history['minute_output_loss'], label='Minute output training loss')
plt.plot(history.history['val_minute_output_loss'], label='Minute output validation loss')
plt.title('Multi-head classification output loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Plot the accuracy curve of hourly output
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['hour_output_accuracy'], label='Hourly output training accuracy')
plt.plot(history.history['val_hour_output_accuracy'], label='Hourly output verification accuracy')
plt.title('Multi-head classification hourly output accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Draw the accuracy curve of minute output
plt.subplot(1, 2, 2)
plt.plot(history.history['minute_output_accuracy'], label='Minute output training accuracy')
plt.plot(history.history['val_minute_output_accuracy'], label='Minute output verification accuracy')
plt.title('Multi-head classification minute output accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Evaluate the model on the test set
results = model.evaluate(X_test, {'hour_output': y_test_hours, 'minute_output': y_test_minutes})
test_loss = results[0]
test_hour_acc = results[3]
test_minute_acc = results[4]

print(f'Test Loss: {test_loss}')
print(f'Test Hour Accuracy: {test_hour_acc}, Test Minute Accuracy: {test_minute_acc}')

# Pick a test image and make predictions
sample_test_image = X_test[0:1]  # Select an image and keep its shape

# Making predictions
hour_pred, minute_pred = model.predict(sample_test_image)

# Get prediction results
predicted_hour = np.argmax(hour_pred)
predicted_minute = np.argmax(minute_pred)
print(f'Predicted Hour: {predicted_hour}, Predicted Minute: {predicted_minute}')

# Display the original image and the predicted results
plt.figure(figsize=(4, 4))
plt.imshow(sample_test_image.squeeze(), cmap='gray')
plt.title(f'Predicted Time: {predicted_hour:02}:{predicted_minute:02}')
plt.axis('off')
plt.show()
