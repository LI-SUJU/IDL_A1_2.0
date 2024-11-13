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
# config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
# sess = tf.Session(config=config)

# 定义超参数
batch_size = 128
epochs = 500
learning_rate = 0.001

# 定义初始化器
kernel_init = HeNormal()  # He Normal 初始化
bias_init = Zeros()  # 偏置初始化为 0

# 加载数据
images = np.load('./data_large/images.npy')  # 替换为实际文件路径
labels = np.load('./data_large/labels.npy')  # 替换为实际文件路径

# images = tf.squeeze(tf.image.resize(tf.expand_dims(images, -1), [299, 299]))

# 归一化图像数据，将像素值缩放到 [0, 1]
images = images / 255.0

# 如果图像是灰度图像，则在最后添加一个维度，以适应 Keras 的输入形状要求
if len(images.shape) == 3:
    images = np.expand_dims(images, axis=-1)  # images = np.repeat(images, 3, axis=-1)

# def preprocess_image(image):
#     images = tf.image.adjust_brightness(image, -0.2)  # 调整亮度
#     images = adjust_contrast(images, contrast_factor=2.5)  # 增强对比度
#     images = adjust_gamma(images, gain=1.0, gamma=4)
#     images = tf.clip_by_value(images, 0.0, 1.0)
#     return images

# images = preprocess_image(images).numpy()

# 将数据集拆分为训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 从标签中分离小时和分钟
y_train_hours = y_train[:, 0]
y_train_minutes = y_train[:, 1]
y_val_hours = y_val[:, 0]
y_val_minutes = y_val[:, 1]
y_test_hours = y_test[:, 0]
y_test_minutes = y_test[:, 1]


# plt.hist(y_train_hours, bins=12)  # 检查小时标签的分布
# plt.hist(y_train_minutes, bins=60)  # 检查分钟标签的分布
# plt.show()
# exit(0)
def build_custom_model(input_shape):
    # 输入层
    inputs = Input(shape=input_shape)
    # 卷积层和池化层
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    # x = layers.BatchNormalization()(x)  # 添加Batch Normalization
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    # x = layers.BatchNormalization()(x)  # 添加Batch Normalization
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    # x = layers.BatchNormalization()(x)  # 添加Batch Normalization
    # # 全局平均池化层
    # x = layers.GlobalAveragePooling2D()(x)
    # 定义输出
    model = models.Model(inputs=inputs, outputs=x)
    return model


input_shape = images[0].shape
base_model = build_custom_model(input_shape=input_shape)
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=None, pooling='avg')
# base_model = VGG16(weights='imagenet', include_top=False)
# 设置保存权重的路径
weights_path = os.path.join(os.getcwd(), 'DIY.weights.h5')
# 保存模型权重
base_model.save_weights(weights_path)
# 定义优化器，使用可调节的学习率
# adam = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# # 冻结层
# for layer in base_model.layers:
#     layer.trainable = False

# 添加decoder
encoder = base_model.output
decoder_1 = Flatten()(encoder)
decoder_1 = Dense(32, activation='relu')(decoder_1)
decoder_1 = Dropout(0.1)(decoder_1)
decoder_hour = Dense(12, activation='sigmoid', name='hour_output')(decoder_1)
decoder_2 = Flatten()(encoder)
decoder_2 = Dense(256, activation='relu')(decoder_2)
decoder_2 = Dropout(0.1)(decoder_2)
decoder_min = Dense(60, activation='sigmoid', name='minute_output')(decoder_2)

# 构建多头模型
model = Model(inputs=base_model.input, outputs=[decoder_hour, decoder_min])


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


    # 如果误差在5分钟以内，将损失视为0
    zero_loss_mask = tf.cast(circular_diff <= 10, tf.float32)

    # 计算插值后的损失值
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    # 结合圆形特性的误差来权重化损失
    weighted_loss = loss * (1 - zero_loss_mask / 60) * (1 - zero_loss_mask)

    return backend.mean(weighted_loss)


# 编译模型
model.compile(optimizer="rmsprop",
              # loss={'hour_output': 'sparse_categorical_crossentropy', 'minute_output': 'sparse_categorical_crossentropy'},
              loss={'hour_output': circular_hour_loss, 'minute_output': circular_minute_loss},
              metrics={'hour_output': 'accuracy', 'minute_output': 'accuracy'})

# 打印模型结构
model.summary()


# 定义数据增强函数
def augment(images, labels):
    images = tf.image.random_brightness(images, 0.2)  # 调整亮度
    images = tf.image.random_contrast(images, 1, 2.0)  # 增强对比度
    images = tf.image.rot90(images, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))  # 随机旋转
    return images, labels  # 返回增强后的图像和原始标签


# 构建 tf.data.Dataset 数据集并应用数据增强
train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, {'hour_output': y_train_hours, 'minute_output': y_train_minutes}))
train_dataset = train_dataset.shuffle(buffer_size=1024).map(augment).batch(batch_size).prefetch(
    tf.data.experimental.AUTOTUNE)

# first_element = next(iter(train_dataset.unbatch()))
# import matplotlib.pyplot as plt
# debug_show_pic = first_element[0]
# # 下面这行是为了转换shape,不需要可以删掉
# plt.imshow(debug_show_pic, cmap='gray')
# plt.show()
# exit(0)


val_dataset = tf.data.Dataset.from_tensor_slices((X_val, {'hour_output': y_val_hours, 'minute_output': y_val_minutes}))
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# 定义回调函数
early_stopping = EarlyStopping(monitor='val_minute_output_accuracy', patience=20, restore_best_weights=True, mode='max')
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_minute_output_accuracy', mode='max')

# 训练模型并加入回调
history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[early_stopping, checkpoint])

model.save('classification_multihead_without_labels_train_model.h5')

# 绘制总损失曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Total training loss')
plt.plot(history.history['val_loss'], label='Validation total loss')
plt.title('Multi-head classification total loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制各输出的损失曲线
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

# 绘制小时输出的准确率曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['hour_output_accuracy'], label='Hourly output training accuracy')
plt.plot(history.history['val_hour_output_accuracy'], label='Hourly output verification accuracy')
plt.title('Multi-head classification hourly output accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 绘制分钟输出的准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['minute_output_accuracy'], label='Minute output training accuracy')
plt.plot(history.history['val_minute_output_accuracy'], label='Minute output verification accuracy')
plt.title('Multi-head classification minute output accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# 在测试集上评估模型
results = model.evaluate(X_test, {'hour_output': y_test_hours, 'minute_output': y_test_minutes})
test_loss = results[0]
test_hour_acc = results[3]
test_minute_acc = results[4]

print(f'Test Loss: {test_loss}')
print(f'Test Hour Accuracy: {test_hour_acc}, Test Minute Accuracy: {test_minute_acc}')

# 选取一个测试图像并进行预测
sample_test_image = X_test[0:1]  # 选取一张图片，保持其形状

# 进行预测
hour_pred, minute_pred = model.predict(sample_test_image)

# 获取预测结果
predicted_hour = np.argmax(hour_pred)
predicted_minute = np.argmax(minute_pred)
print(f'Predicted Hour: {predicted_hour}, Predicted Minute: {predicted_minute}')

# 显示原图及预测结果
plt.figure(figsize=(4, 4))
plt.imshow(sample_test_image.squeeze(), cmap='gray')
plt.title(f'Predicted Time: {predicted_hour:02}:{predicted_minute:02}')
plt.axis('off')
plt.show()
