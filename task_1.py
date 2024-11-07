import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.initializers import RandomNormal, GlorotUniform, LecunNormal
from tensorflow.keras.regularizers import l1, l2

# Load and preprocess Fashion MNIST data
(fashion_x_train, fashion_y_train), (fashion_x_test, fashion_y_test) = fashion_mnist.load_data()
fashion_x_train, fashion_x_test = fashion_x_train / 255.0, fashion_x_test / 255.0
fashion_x_train, fashion_x_val, fashion_y_train, fashion_y_val = train_test_split(
    fashion_x_train, fashion_y_train, test_size=0.1, random_state=42
)
fashion_y_train = to_categorical(fashion_y_train)
fashion_y_val = to_categorical(fashion_y_val)
fashion_y_test = to_categorical(fashion_y_test)

# Load and preprocess CIFAR-10 data
(cifar_x_train, cifar_y_train), (cifar_x_test, cifar_y_test) = cifar10.load_data()
cifar_x_train, cifar_x_test = cifar_x_train / 255.0, cifar_x_test / 255.0
cifar_x_train, cifar_x_val, cifar_y_train, cifar_y_val = train_test_split(
    cifar_x_train, cifar_y_train, test_size=0.1, random_state=42
)
cifar_y_train = to_categorical(cifar_y_train)
cifar_y_val = to_categorical(cifar_y_val)
cifar_y_test = to_categorical(cifar_y_test)

# Initialize weights with HeNormal
initializer = HeNormal()

# Initialize weights with different initializers as shown below
random_normal_initializer = RandomNormal(mean=0.0, stddev=0.05)
glorot_uniform_initializer = GlorotUniform()
lecun_normal_initializer = LecunNormal()

# activation functions to be used in the model
activation_functions = {
    'relu': 'relu',
    'softmax': 'softmax',
    'sigmoid': 'sigmoid',
    'tanH': 'tanh'
}

# make a list of optimizers to be used in the model and include the hyperparameters of them
optimizers = {
    'adam': 'adam',
    'rmsprop': 'rmsprop',
    'sgd': 'sgd'
}

# adjust the learning rate of the optimizer
adam_optimizer = optimizers['adam']
adam_optimizer = optimizers['adam'](learning_rate=0.01)

# make a list of loss functions to be used in the model
loss_functions = {
    'categorical_crossentropy': 'categorical_crossentropy',
    'mean_squared_error': 'mean_squared_error',
    'mean_absolute_error': 'mean_absolute_error'
}

# regularizers to be used in the model, L1, L2, and dropout
# Use the regularizers in the model mlp_model:
# mlp_model = Sequential([
#     Flatten(input_shape=(28, 28)),
#     Dense(512, activation='relu', kernel_initializer=initializer, kernel_regularizer=l1(0.01)),
#     Dropout(0.2),
#     Dense(10, activation='softmax', kernel_initializer=initializer)
# ])

# Build MLP model for Fashion MNIST
mlp_model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu', kernel_initializer=initializer),
    Dropout(0.2),
    Dense(10, activation='softmax', kernel_initializer=initializer)
])
mlp_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train MLP model
mlp_history = mlp_model.fit(
    fashion_x_train, fashion_y_train,
    validation_data=(fashion_x_val, fashion_y_val),
    epochs=10,
    batch_size=32
)

# Build CNN model for Fashion MNIST
fashion_cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

fashion_cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train CNN model for Fashion MNIST
fashion_cnn_history = fashion_cnn_model.fit(
    fashion_x_train[..., np.newaxis], fashion_y_train,
    validation_data=(fashion_x_val[..., np.newaxis], fashion_y_val),
    epochs=10,
    batch_size=32
)

# Build CNN model for CIFAR-10
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train CNN model for CIFAR-10
cnn_history = cnn_model.fit(
    cifar_x_train, cifar_y_train,
    validation_data=(cifar_x_val, cifar_y_val),
    epochs=10,
    batch_size=32
)

# Plot results
def plot_history(history, title):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title(f'{title} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title(f'{title} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_history(mlp_history, 'MLP Fashion MNIST')
plot_history(fashion_cnn_history, 'CNN Fashion MNIST')
plot_history(cnn_history, 'CNN CIFAR-10')