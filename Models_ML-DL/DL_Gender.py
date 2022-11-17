# Conjuntos de dados: https://www.kaggle.com/ashishjangra27/gender-recognition-200k-images-celeba/download

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import image
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam


# ler e importar o conjunto de dados de imagens que vamos usar para treinar um modelo de rede neural

train_datagen = ImageDataGenerator(
    rescale = 1./255,
	rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest')


test_datagen = ImageDataGenerator(rescale = 1.0/255)

train_generator = train_datagen.flow_from_directory(
    "gender-recognition-200k-images-celeba/Dataset/Train/",
    batch_size = 256 ,
    class_mode = 'binary', 
    target_size = (64, 64))

model = tf.keras.models.Sequential([

    # 1st conv

  tf.keras.layers.Conv2D(96, (11,11),strides=(4,4), activation='relu', input_shape=(64, 64, 3)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2, strides=(2,2)),

    # 2nd conv

  tf.keras.layers.Conv2D(256, (11,11),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),

     # 3rd conv

  tf.keras.layers.Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),

    # 4th conv

  tf.keras.layers.Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),

    # 5th Conv

  tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2, strides=(2, 2)),

  # To Flatten layer

  tf.keras.layers.Flatten(),

  # To FC layer 1

  tf.keras.layers.Dense(4096, activation='relu'),
  tf.keras.layers.Dropout(0.5),

  #To FC layer 2

  tf.keras.layers.Dense(4096, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(1, activation='sigmoid')
  ])

model.compile(
    optimizer = Adam(lr=0.001),
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
   )

hist = model.fit_generator(
    generator = train_generator,
    validation_data = validation_generator,
    steps_per_epoch = 256,
    validation_steps = 256,
    epochs = 50)


validation_generator =  test_datagen.flow_from_directory( 
    "gender-recognition-200k-images-celeba/Dataset/Validation/",
    batch_size  = 256,
    class_mode  = 'binary', 
    target_size = (64, 64))

# desempenho do modelo em termos de precisão

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label = 'Training accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc = 0)
plt.figure()
plt.show()

# testar o modelo de rede neural
# prever imagens

path = "gender-recognition-200k-images-celeba/Dataset/Test/Female/160001.jpg"
img = image.load_img(path, target_size = (64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

images = np.vstack([x])
classes = model.predict(images, batch_size = 1)
print(classes[0])

if classes[0] > 0.5:
    print("is a man")
else:
    print( " is a female")
plt.imshow(img)