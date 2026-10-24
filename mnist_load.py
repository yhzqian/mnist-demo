# mnist_keras.py
import tensorflow as tf
import matplotlib.pyplot as plt

# ========== 1. 加载数据 ==========
# 注意：这里用的是 tf.keras 自带的数据加载，不是 torchvision.datasets
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("训练集形状:", x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
print("测试集形状:", x_test.shape, y_test.shape)    # (10000, 28, 28) (10000,)

# ========== 2. 预处理 ==========
# 像素值从 0~255 归一化到 0~1，加快训练
x_train = x_train / 255.0
x_test = x_test / 255.0

# ========== 3. 构建模型 ==========
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),      # 28x28 拉平成 784
    tf.keras.layers.Dense(128, activation='relu'),      # 隐藏层
    tf.keras.layers.Dropout(0.2),                       # 防过拟合
    tf.keras.layers.Dense(10, activation='softmax')     # 输出 10 个类别概率
])

model.summary()

# ========== 4. 编译 ==========
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ========== 5. 训练 ==========
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,   # 从训练集里划 10% 做验证
    verbose=2
)

# ========== 6. 评估 ==========
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n测试集准确率: {test_acc:.4f}")

# ========== 7. 预测单张图片（可选，直观感受） ==========
import numpy as np

idx = 0
img = x_test[idx]
true_label = y_test[idx]

pred = model.predict(img.reshape(1, 28, 28), verbose=0)
pred_label = np.argmax(pred)

plt.imshow(img, cmap='gray')
plt.title(f"真实: {true_label}  预测: {pred_label}")
plt.axis('off')
plt.show()

print(f"真实标签: {true_label}, 预测标签: {pred_label}")
print(f"各类别概率: {pred}")