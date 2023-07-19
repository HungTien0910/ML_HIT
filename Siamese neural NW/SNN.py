import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Tạo bộ dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Chuẩn hóa và chuyển đổi dữ liệu về dạng float32
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Hàm lấy mẫu ngẫu nhiên để tạo cặp dữ liệu huấn luyện
def get_random_pair(x, y):
    class_indices = [np.where(y == i)[0] for i in range(10)]
    anchor_class = np.random.randint(0, 10)
    positive_class = anchor_class
    while positive_class == anchor_class:
        positive_class = np.random.randint(0, 10)

    anchor_idx = np.random.choice(class_indices[anchor_class])
    positive_idx = np.random.choice(class_indices[positive_class])

    return x[anchor_idx], x[positive_idx]

# Tạo dữ liệu huấn luyện
def generate_data(x, y, batch_size=64):
    while True:
        x_anchor, x_positive = [], []
        for _ in range(batch_size):
            anchor, positive = get_random_pair(x, y)
            x_anchor.append(anchor)
            x_positive.append(positive)
        yield [np.array(x_anchor), np.array(x_positive)], np.zeros(batch_size)

# Xây dựng Siamese Neural Network
def build_siamese_model(input_shape):
    input_layer = keras.Input(shape=input_shape)
    x = layers.Flatten()(input_layer)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    output_layer = layers.Dense(10, activation="sigmoid")(x)
    return keras.Model(input_layer, output_layer)

input_shape = x_train[0].shape
model = build_siamese_model(input_shape)

# Tạo mô hình Siamese với đầu vào là cặp hình ảnh
input_anchor = keras.Input(shape=input_shape)
input_positive = keras.Input(shape=input_shape)

output_anchor = model(input_anchor)
output_positive = model(input_positive)

# Hàm tính khoảng cách Euclidean giữa các vector đầu ra của cặp hình ảnh
def euclidean_distance(vectors):
    x, y = vectors
    sum_squared = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_squared, tf.keras.backend.epsilon()))

# Kết hợp mô hình thành mô hình Siamese hoàn chỉnh
distance = layers.Lambda(euclidean_distance)([output_anchor, output_positive])
siamese_model = keras.Model(inputs=[input_anchor, input_positive], outputs=distance)

# Tạo hàm loss function
def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        loss = y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
        return tf.reduce_mean(loss)

    return contrastive_loss

# Compile mô hình
margin = 1.0
siamese_model.compile(optimizer="adam", loss=contrastive_loss_with_margin(margin))

# Huấn luyện mô hình
batch_size = 64
epochs = 10
siamese_model.fit(generate_data(x_train, y_train, batch_size), steps_per_epoch=len(x_train) // batch_size, epochs=epochs)

# Đánh giá mô hình
def get_test_pair(x, y, class_idx):
    class_indices = np.where(y == class_idx)[0]
    idx1, idx2 = np.random.choice(class_indices, 2, replace=False)
    return x[idx1], x[idx2]

def evaluate_model(model, x, y, num_trials=100):
    correct = 0
    for _ in range(num_trials):
        class_idx = np.random.randint(0, 10)
        x1, x2 = get_test_pair(x, y, class_idx)
        x1 = np.expand_dims(x1, axis=0)
        x2 = np.expand_dims(x2, axis=0)
        distance = model.predict([x1, x2])[0][0]
        if distance < margin:
            correct += 1

    accuracy = correct / num_trials
    return accuracy

accuracy = evaluate_model(siamese_model, x_test, y_test)
print("Accuracy:", accuracy)
