import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Input, Model, backend
from sklearn.model_selection import train_test_split
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load a smaller subset of your data
#####################################################################
#####################################################################
#####################################################################
# 请在下面两行代码调整样本量
#####################################################################
#####################################################################
#####################################################################
# randomly sample 500 images

images = np.load('./data/images.npy')  # Use a subset of data
labels = np.load('./data/labels.npy')
images = images / 255.0
if len(images.shape) == 3:
    images = np.expand_dims(images, axis=-1)

X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

y_train_hours = y_train[:, 0]
y_train_minutes = y_train[:, 1]
y_val_hours = y_val[:, 0]
y_val_minutes = y_val[:, 1]

def build_model(config):
    input_shape = X_train[0].shape
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(config["conv1_units"], (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(config["conv2_units"], (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    encoder = x
    decoder_1 = layers.Flatten()(encoder)
    decoder_1 = layers.Dense(config["dense1_units"], activation='relu')(decoder_1)
    decoder_1 = layers.Dropout(config["dropout"])(decoder_1)
    decoder_hour = layers.Dense(12, activation='softmax', name='hour_output')(decoder_1)

    decoder_2 = layers.Flatten()(encoder)
    decoder_2 = layers.Dense(config["dense2_units"], activation='relu')(decoder_2)
    decoder_2 = layers.Dropout(config["dropout"])(decoder_2)
    decoder_min = layers.Dense(60, activation='softmax', name='minute_output')(decoder_2)

    model = Model(inputs=inputs, outputs=[decoder_hour, decoder_min])
    return model

class TuneReportCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Report metrics to Ray Tune
        session.report({
            "hour_accuracy": logs.get("val_hour_output_accuracy"),
            "minute_accuracy": logs.get("val_minute_output_accuracy")
        })



# exit(0)


def circular_hour_loss(y_true, y_pred):
    # 将 y_true 处理为整数标签
    y_true = tf.cast(y_true, tf.float32)

    # 找到最可能的预测类别
    pred_classes = tf.argmax(y_pred, axis=-1)
    pred_classes = tf.cast(pred_classes, tf.float32)

    # 计算正向误差和反向误差（12小时制）
    forward_diff = tf.abs(pred_classes - y_true)
    backward_diff = 12 - forward_diff

    # 根据最小误差来进行插值，最小误差部分较多的分类将获得更大的权重
    circular_diff = tf.minimum(forward_diff, backward_diff)

    # 计算插值后的损失值
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    # 结合圆形特性的误差来权重化损失
    weighted_loss = loss * (1 - circular_diff / 12)

    return backend.mean(weighted_loss)


def circular_minute_loss(y_true, y_pred):
    # 将 y_true 处理为整数标签
    y_true = tf.cast(y_true, tf.float32)

    # 找到最可能的预测类别
    pred_classes = tf.argmax(y_pred, axis=-1)
    pred_classes = tf.cast(pred_classes, tf.float32)

    # 计算正向误差和反向误差（12小时制）
    forward_diff = tf.abs(pred_classes - y_true)
    backward_diff = 60 - forward_diff

    # 根据最小误差来进行插值，最小误差部分较多的分类将获得更大的权重
    circular_diff = tf.minimum(forward_diff, backward_diff)


    # 如果误差在10分钟以内，将损失视为0
    zero_loss_mask = tf.cast(circular_diff <= 10, tf.float32)

    # 计算插值后的损失值
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    # 结合圆形特性的误差来权重化损失
    weighted_loss = loss * (1 - zero_loss_mask / 60) * (1 - zero_loss_mask)

    return backend.mean(weighted_loss)

def train_model(config, X_train=None, y_train_hours=None, y_train_minutes=None, X_val=None, y_val_hours=None, y_val_minutes=None):
    model = build_model(config)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
                  loss={'hour_output': circular_hour_loss,
                        'minute_output': circular_minute_loss},
                  metrics={'hour_output': 'accuracy', 'minute_output': 'accuracy'})

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train, {'hour_output': y_train_hours, 'minute_output': y_train_minutes}))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(config["batch_size"]).prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (X_val, {'hour_output': y_val_hours, 'minute_output': y_val_minutes}))
    val_dataset = val_dataset.batch(config["batch_size"]).prefetch(tf.data.experimental.AUTOTUNE)

    # Use the custom callback
    callbacks = [TuneReportCallback()]

    model.fit(train_dataset,
              validation_data=val_dataset,
              epochs=5,  # Use fewer epochs
              callbacks=callbacks)

#####################################################################
#####################################################################
#####################################################################
# 请在下面的字典中调整超参搜索空间，空间越大，搜索得越全面，时间也会更长
#####################################################################
#####################################################################
#####################################################################
search_space = {
    "lr": tune.loguniform(1e-5, 1e-2),
    "conv1_units": tune.choice([16, 32, 64]),  # Reduced options
    "conv2_units": tune.choice([16, 32, 64]),  # Reduced options
    "dense1_units": tune.choice([32, 64, 128]),  # Reduced options
    "dense2_units": tune.choice([64, 128, 256]),  # Reduced options
    "dropout": tune.uniform(0.1, 0.5),
    "batch_size": tune.choice([16, 32, 64])  # Smaller batch size
}

scheduler = ASHAScheduler(
    # metric="minute_accuracy",
    # mode="max",
    max_t=5,  # Reduced max training time
    grace_period=1,
    reduction_factor=2)

# Define a valid directory path for storage
storage_dir = './ray_results'

# Ensure the directory exists
os.makedirs(storage_dir, exist_ok=True)

analysis = tune.run(
    tune.with_parameters(train_model,
                         X_train=X_train,
                         y_train_hours=y_train_hours,
                         y_train_minutes=y_train_minutes,
                         X_val=X_val,
                         y_val_hours=y_val_hours,
                         y_val_minutes=y_val_minutes),
    resources_per_trial={"cpu": 1, "gpu": 0},  # Use fewer resources
    config=search_space,
    num_samples=5,  # Fewer trials
    scheduler=scheduler,
    # local_dir=storage_dir,  # Specify the local directory for results
    metric="minute_accuracy",  # Specify the metric to optimize
    mode="max"  # Specify whether to maximize or minimize the metric
)

# Extract the best configurations and accuracies
best_config = analysis.get_best_config(metric="minute_accuracy", mode="max")
best_trial = analysis.get_best_trial(metric="minute_accuracy", mode="max")
best_hour_accuracy = best_trial.metric_analysis["hour_accuracy"]["max"]
best_minute_accuracy = best_trial.metric_analysis["minute_accuracy"]["max"]

print("Best hyperparameters found were: ", best_config)
print(f"Best hour accuracy: {best_hour_accuracy}")
print(f"Best minute accuracy: {best_minute_accuracy}")

