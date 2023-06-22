import tensorflow as tf
import matplotlib.pyplot as plt

# Load and preprocess the image
image_path = 'DL\main_image_url1-HW-HW03142303.jpeg'
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
image = tf.keras.preprocessing.image.img_to_array(image)
image = tf.expand_dims(image, axis=0)
image /= 255.0

# Define the model with Convolution, Padding, and Pooling layers
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Visualize the layers
layer_names = ['Convolution Layer', 'Padding Layer', 'Pooling Layer']
layer_outputs = [model.layers[0].output, model.layers[1].output, model.layers[2].output]

for i, output in enumerate(layer_outputs):
    intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=output)
    intermediate_output = intermediate_model.predict(image)
    intermediate_output = intermediate_output[0]

    plt.figure(figsize=(8, 8))
    for j in range(intermediate_output.shape[-1]):
        plt.subplot(8, 8, j + 1)
        plt.imshow(intermediate_output[:, :, j], cmap='gray')
        plt.axis('off')
    plt.suptitle(layer_names[i])
    plt.show()
