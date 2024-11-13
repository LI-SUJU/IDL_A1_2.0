import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 加载数据
images = np.load('./data/data_big/images.npy')
labels = np.load('./data/data_big/labels.npy')

# 归一化图像数据，将像素值缩放到 [0, 1]
images = images / 255.0

# 如果图像是灰度图像，添加一个维度
if len(images.shape) == 3:
    images = np.expand_dims(images, axis=-1)

# 将标签分为小时和分钟
hours = labels[:, 0]
minutes = labels[:, 1]

# 检查样本数量是否一致
print("Number of images:", images.shape[0])
print("Number of labels:", labels.shape[0])

# 划分数据集
# Stack hours and minutes for simultaneous split, then separate them afterward
labels_combined = np.stack([hours, minutes], axis=1)
# First split into train and val+test
X_train, X_val_test, y_train_combined, y_val_test_combined = train_test_split(
    images, labels_combined, test_size=0.2, random_state=42
)

# Then split val+test into val and test
X_val, X_test, y_val_combined, y_test_combined = train_test_split(
    X_val_test, y_val_test_combined, test_size=0.5, random_state=42
)

# Separate hours and minutes after split
y_train_hours, y_train_minutes = y_train_combined[:, 0], y_train_combined[:, 1]
y_val_hours, y_val_minutes = y_val_combined[:, 0], y_val_combined[:, 1]
y_test_hours, y_test_minutes = y_test_combined[:, 0], y_test_combined[:, 1]



# 构建 CNN 模型
input_img = Input(shape=(150, 150, 1))
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_img)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Flatten()(x)
x = Dense(64, activation='relu')(x)

# 两个输出节点，分别用于输出小时和分钟
hour_output = Dense(1, activation='linear', name='hour_output')(x)
minute_output = Dense(1, activation='linear', name='minute_output')(x)

# 创建模型
model = Model(inputs=input_img, outputs=[hour_output, minute_output])

# 编译模型，使用均方误差损失和平均绝对误差指标
model.compile(optimizer=Adam(learning_rate=0.001),
              loss={'hour_output': 'mse', 'minute_output': 'mse'},
              metrics={'hour_output': 'mae', 'minute_output': 'mae'})

# 训练模型
history = model.fit(
    X_train, {'hour_output': y_train_hours, 'minute_output': y_train_minutes},
    epochs=100,
    batch_size=16,
    validation_data=(X_test, {'hour_output': y_test_hours, 'minute_output': y_test_minutes})
)

#save model
model.save('two_head_regressor.h5')

# 评估模型
test_loss, test_hour_loss, test_minute_loss, test_hour_mae, test_minute_mae = model.evaluate(
    X_test, {'hour_output': y_test_hours, 'minute_output': y_test_minutes}, verbose=2)
print(f"Test Hour Loss: {test_hour_loss}")
print(f"Test Minute Loss: {test_minute_loss}")
print(f"Test Hour MAE: {test_hour_mae}")
print(f"Test Minute MAE: {test_minute_mae}")

# 绘制训练曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['hour_output_loss'], label='Training Hour Loss')
plt.plot(history.history['val_hour_output_loss'], label='Validation Hour Loss')
plt.plot(history.history['minute_output_loss'], label='Training Minute Loss')
plt.plot(history.history['val_minute_output_loss'], label='Validation Minute Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['hour_output_mae'], label='Training Hour MAE')
plt.plot(history.history['val_hour_output_mae'], label='Validation Hour MAE')
plt.plot(history.history['minute_output_mae'], label='Training Minute MAE')
plt.plot(history.history['val_minute_output_mae'], label='Validation Minute MAE')
plt.title('MAE Curve')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.savefig('two_head_regressor.png')

# 预测时间
predictions = model.predict(X_test[:5])
for i, (image, hour, minute) in enumerate(zip(X_test[:5], predictions[0], predictions[1])):
    hour_value = int(hour[0])
    minute_value = int(minute[0])

    # 打印预测的时间
    print(f"Predicted time for sample {i + 1}: {hour_value}:{minute_value:02d}")

    # 显示图像
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Predicted Time: {hour_value}:{minute_value:02d}")
    plt.axis('off')
    plt.show()
