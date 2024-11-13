import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.initializers import HeNormal, Zeros
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 超参数
batch_size = 128
epochs = 500
learning_rate = 0.0001

# 初始化器
kernel_init = HeNormal()
bias_init = Zeros()

# 加载数据
images = np.load('./data_large/images.npy')  # 替换为实际文件路径
labels = np.load('./data_large/labels.npy')  # 替换为实际文件路径

# 归一化图像数据
images = images / 255.0

# 图像形状
if len(images.shape) == 3:
    images = np.expand_dims(images, axis=-1)

# 标签转换为小数形式
def convert_labels_to_decimal(labels):
    hours = labels[:, 0]
    minutes = labels[:, 1]
    decimal_time = hours + minutes / 60.0
    return decimal_time

decimal_labels = convert_labels_to_decimal(labels)

# 数据集划分
X_train, X_temp, y_train, y_temp = train_test_split(images, decimal_labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 自定义模型
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
    outputs = Dense(1, activation='linear')(x)  # 输出1个连续值
    model = Model(inputs=inputs, outputs=outputs)
    return model

input_shape = images[0].shape
model = build_custom_model(input_shape=input_shape)

# 自定义损失函数，考虑周期性差异
def time_difference(y_true, y_pred):
    diff = tf.abs(y_true - y_pred)
    return tf.minimum(diff, 12 - diff)  # 假设时间在12小时制内循环

def custom_loss(y_true, y_pred):
    diff = time_difference(y_true, y_pred)
    return tf.reduce_mean(tf.square(diff))

# 自定义 MAE 指标，考虑周期性差异
def custom_mae(y_true, y_pred):
    diff = time_difference(y_true, y_pred)
    return tf.reduce_mean(diff)

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=custom_loss,
    metrics=[custom_mae]  # 使用自定义的 MAE 指标
)

# 打印模型结构
model.summary()

# 定义数据增强函数
def augment(images, labels):
    images = tf.image.random_brightness(images, 0.2)
    images = tf.image.random_contrast(images, 1, 2.0)
    images = tf.image.rot90(images, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    return images, labels

# 构建 tf.data.Dataset 数据集并应用数据增强
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).map(augment).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# 定义回调函数
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, mode='min')
checkpoint = ModelCheckpoint('best_Regression_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# 训练模型
history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[early_stopping, checkpoint])

# 设置保存权重的路径
weights_path = os.path.join(os.getcwd(), 'Regression.weights.h5')
# 保存模型权重
model.save_weights(weights_path)

# 在测试集上评估模型
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")

# 预测示例
sample_test_image = X_test[0:1]
predicted_output = model.predict(sample_test_image)

# 将预测值转换回小时和分钟
def convert_prediction_to_time(pred):
    pred_time = pred[0] % 12  # 确保在 0 到 12 之间
    if pred_time < 0:
        pred_time += 12  # 确保时间为正数
    pred_hour = int(pred_time)
    pred_minute = int(round((pred_time - pred_hour) * 60)) % 60
    return pred_hour % 12, pred_minute

predicted_hour, predicted_minute = convert_prediction_to_time(predicted_output[0])
print(f'Predicted Time: {predicted_hour:02}:{predicted_minute:02}')

# 显示原图及预测结果
plt.figure(figsize=(4, 4))
plt.imshow(sample_test_image.squeeze(), cmap='gray')
plt.title(f'Predicted Time: {predicted_hour:02}:{predicted_minute:02}')
plt.axis('off')
plt.show()

# **绘制训练曲线**
# 绘制训练和验证的损失曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制训练和验证的自定义 MAE 曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['custom_mae'], label='Training MAE')
plt.plot(history.history['val_custom_mae'], label='Validation MAE')
plt.title('Custom MAE Curve')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.show()