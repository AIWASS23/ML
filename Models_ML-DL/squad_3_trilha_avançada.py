
# Commented out IPython magic to ensure Python compatibility.
import os
import glob
import zipfile
import cv2 as cv
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import keras.layers as layers
 
 
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')

"""##Funções básicas

###Funcionalidade para leitura de arquivos
"""

def readDir(diretorio):
    """
        Função para listagem de imagens
        :param diretorio: enedereço do diretório
    """
    listaimgs = os.listdir(diretorio)
    return listaimgs

def readImage(local, listaimgs):
    """
        Função para leitura de imagens
        :param local: pasta das imagens
        :param listaimgs: nomes das imagens
    """
    imagens = []
    for i in listaimgs:
        img = cv.imread(local+i)
        img = cv.resize(img, (196, 196))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = img/255.0
        img = np.reshape(img, (196,196,1))
        imagens.append(img)
    
    return imagens

"""###Funcionalidade para exibir imagens"""

def mostrar_imagemRGB(imagem):
  imagem_rgb = cv.cvtColor(imagem, cv.COLOR_BGR2RGB)
  plt.imshow(imagem_rgb)

def mostrar_imagemGRAY(imagem):
  imagem_gray = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)
  plt.imshow(imagem_gray)

"""###Fucionalidade para aplicar LBP"""

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y-1))
    val_ar.append(get_pixel(img, center, x-1, y))
    val_ar.append(get_pixel(img, center, x-1, y + 1))
    val_ar.append(get_pixel(img, center, x, y + 1))
    val_ar.append(get_pixel(img, center, x + 1, y + 1))
    val_ar.append(get_pixel(img, center, x + 1, y))
    val_ar.append(get_pixel(img, center, x + 1, y-1))
    val_ar.append(get_pixel(img, center, x, y-1))
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val

# img = pipeline(amostras[21])
# img_lbp = np.zeros((len(img[0]), len(img)), np.uint8)
# for i in range(0, len(img[0])):
#   for j in range(0, len(img)):
#       img_lbp[i, j] = lbp_calculated_pixel(img, i, j)
# mostrar_imagemRGB(img_lbp)
# print(classe[21])

"""###Funcionalidade para aplicar Adaptive Threshold"""

def adaptiveThresh(imagem):
    imagem = cv.resize(imagem, (196, 196))
    imagem = imagem[0:170, :]
    imagem = cv.equalizeHist(imagem)
    imagem = cv.GaussianBlur(imagem, (5,5), 0)
    thresh1 = cv.adaptiveThreshold(imagem, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 199, 5)
    thresh2 = cv.adaptiveThreshold(imagem, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 199, 5)
    return thresh2

"""###Funcionalidade para plotagem da matriz de confusão"""

def get_confusion_matrix(y_true, y_pred, model):
    plt.subplots(figsize=(8,6))
    labels = ['Normal','Covid','default']
    cm = metrics.confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', cmap='YlGnBu').set(title=model, ylabel='Real', xlabel='Predito', 
                                                            yticklabels=labels,
                                                            xticklabels=labels)

# Importa as funções necessárias
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc

# Função para gerar relatório de métricas
def print_report(y_actual, y_pred, thresh):
    
    mdc = confusion_matrix(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    precisao = precision_score(y_actual, y_pred)
    sensibil = recall_score(y_actual, y_pred)
    fpr, tpr, _ = roc_curve(y_actual, y_pred)
    roc_auc = auc(fpr, tpr)
    print(mdc)
    print('Acurácia:%.3f'%accuracy)
    print('Precisão:%.3f'%precisao)
    print('Sensibilidade:%.3f'%sensibil)
    print(' ')
    return mdc, accuracy, precisao, sensibil, fpr, tpr, roc_auc

"""##Lendo base de dados"""

endereco = '/content/drive/MyDrive/dataset.zip'

zip_object = zipfile.ZipFile(file=endereco, mode='r')
zip_object.extractall('./') 
zip_object.close

"""##Lendo imagens da base de dados"""

dataset = 'dataset/imagens/'

imagens = readDir(dataset)
img = readImage(dataset, imagens)

def carrega_data(listaimgs, imagens):
  i = 0
  amostras = []
  classe = []
  for img in listaimgs:
    amostras.append(imagens[i])
    clas = img.split('_')
    classe.append(clas[1][0])
    i = i + 1
  return amostras, classe

amostras, classe = carrega_data(imagens, img)

"""##Dividindo dados de treino e teste"""

X_train, X_test, y_train, y_test = train_test_split(amostras, classe, train_size=0.3, random_state=42)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = to_categorical(y_train)

# define generator
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range = 0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)

# fit generator on our train features
datagen.fit(X_train)
datagen.fit(X_test)

"""##Expandindo dados de treino e teste

##Modelos de redes CNN

###AlexNet
"""

def alex():
  model = Sequential()

  model.add(Conv2D(filters=128, kernel_size=(11,11), strides=(4,4), padding="same", activation="relu", input_shape=(196,196,1)))
  model.add(BatchNormalization())
  model.add(MaxPool2D(pool_size=(2,2)))

  model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu"))
  model.add(BatchNormalization())
  model.add(MaxPool2D(pool_size=(3,3)))

  model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
  model.add(BatchNormalization())

  model.add(Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu"))
  model.add(BatchNormalization())

  model.add(Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu"))
  model.add(BatchNormalization())
  model.add(MaxPool2D(pool_size=(2,2)))

  model.add(Flatten())

  model.add(Dense(units=1024,activation="relu"))
  model.add(Dropout(0.5))
  model.add(Dense(units=1024,activation="relu"))
  model.add(Dropout(0.5))
  model.add(Dense(units=2,activation="softmax"))

  optimizer = Adam(lr=0.0001, decay=1e-5)
  model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
  return model

"""###Inception V3"""

from keras.layers import Dense, Dropout, Flatten, Activation

def v3():
  model = Sequential()

  model.add(Conv2D(32, (3, 3), input_shape=(196, 196, 1)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Dropout(0.25))

  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(256))
  model.add(Activation('relu'))

  model.add(Dropout(0.5))

  model.add(Dense(2))
  model.add(Activation('softmax'))

  model.summary()

  model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])
  return model

"""###LeNet"""

def le():
  model = Sequential()

  model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(196,196,1)))
  model.add(layers.AveragePooling2D())

  model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
  model.add(layers.AveragePooling2D())

  model.add(layers.Flatten())

  model.add(layers.Dense(units=128, activation='relu'))

  model.add(layers.Dense(units=84, activation='relu'))

  model.add(layers.Dense(units=2, activation = 'softmax'))

  optimizer = Adam(lr=0.0001, decay=1e-5)
  model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
  return model

"""##Metodologias

###Metodologia 1


*   Imagem original
*   Aplicações de CNN
"""

model_alex = alex()

callback = EarlyStopping(monitor='loss', patience=5, mode='min')
history = model_alex.fit(X_train, y_train, batch_size=1, epochs = 50, verbose = 1, callbacks=[callback])

model_alex.summary()

model_alex.save(filepath='modelalexnet.hdf5')

fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(121)
ax.plot(history.history["loss"])
ax.set_title("validation loss")
ax.set_xlabel("epochs")

ax2 = fig.add_subplot(122)
ax2.plot(history.history["accuracy"])
ax2.set_title("validation accuracy")
ax2.set_xlabel("epochs")
ax2.set_ylim(0, 1)

plt.show()

model_le = le()

callback = EarlyStopping(monitor='loss', patience=5, mode='min')
history = model_le.fit(X_train, y_train, batch_size=1, epochs = 50, verbose = 1, callbacks=[callback])

model_le.summary()

model_le.save(filepath='modellenet.hdf5')

fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(121)
ax.plot(history.history["loss"])
ax.set_title("validation loss")
ax.set_xlabel("epochs")

ax2 = fig.add_subplot(122)
ax2.plot(history.history["accuracy"])
ax2.set_title("validation accuracy")
ax2.set_xlabel("epochs")
ax2.set_ylim(0, 1)

plt.show()

"""###Metodologia 2


*   Imagem pré-processada com filtro gaussiano
*   Aplicações de CNN
"""

def carrega_data(listaimgs, imagens):
  i = 0
  amostras = []
  classe = []
  for img in listaimgs:
    imag = cv.resize(imagens[i], (196, 196))
    imag = imag/255.0
    imag = np.reshape(imag, (196,196,1))
    imag= cv.GaussianBlur(imag, (5,5), 0)
    amostras.append(imag)
    clas = img.split('_')
    classe.append(clas[1][0])
    i = i + 1
  return amostras, classe

amostras, classe = carrega_data(imagens, img)

X_train, X_test, y_train, y_test = train_test_split(amostras, classe, train_size=0.3, random_state=42)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = to_categorical(y_train)

# define generator
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range = 0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)

# # fit generator on our train features
# datagen.fit(X_train)
# datagen.fit(X_test)

model_alex = alex()

callback = EarlyStopping(monitor='loss', patience=5, mode='min')
history = model_alex.fit(X_train, y_train, batch_size=1, epochs = 50, verbose = 1, callbacks=[callback])

model_alex.summary()

model_alex.save(filepath='modelalexnet.hdf5')

fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(121)
ax.plot(history.history["loss"])
ax.set_title("validation loss")
ax.set_xlabel("epochs")

ax2 = fig.add_subplot(122)
ax2.plot(history.history["accuracy"])
ax2.set_title("validation accuracy")
ax2.set_xlabel("epochs")
ax2.set_ylim(0, 1)

plt.show()

def carrega_data(listaimgs, imagens):
  i = 0
  amostras = []
  classe = []
  for img in listaimgs:
    imag = cv.resize(imagens[i], (196, 196))
    imag = imag/255.0
    imag = np.reshape(imag, (196,196,1))
    imag= cv.GaussianBlur(imag, (5,5), 0)
    amostras.append(imag)
    clas = img.split('_')
    classe.append(clas[1][0])
    i = i + 1
  return amostras, classe

amostras, classe = carrega_data(imagens, img)

X_train, X_test, y_train, y_test = train_test_split(amostras, classe, train_size=0.3, random_state=42)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# define generator
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range = 0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)

# # fit generator on our train features
# datagen.fit(X_train)
# datagen.fit(X_test)

model_le = le()

callback = EarlyStopping(monitor='loss', patience=5, mode='min')
history = model_le.fit(X_train, y_train, batch_size=1, epochs = 50, verbose = 1, callbacks=[callback])

model_le.summary()

model_le.save(filepath='modellenet.hdf5')

fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(121)
ax.plot(history.history["loss"])
ax.set_title("validation loss")
ax.set_xlabel("epochs")

ax2 = fig.add_subplot(122)
ax2.plot(history.history["accuracy"])
ax2.set_title("validation accuracy")
ax2.set_xlabel("epochs")
ax2.set_ylim(0, 1)

plt.show()

"""###Metodologia 3


*   Imagem pré-processada com binarização
*   Aplicações de CNN
"""

def carrega_data(listaimgs, imagens):
  i = 0
  amostras = []
  classe = []
  for img in listaimgs:
    imag = adaptiveThresh(imagens[i])
    amostras.append(imag)
    clas = img.split('_')
    classe.append(clas[1][0])
    i = i + 1
  return amostras, classe

amostras, classe = carrega_data(imagens, img)

X_train, X_test, y_train, y_test = train_test_split(amostras, classe, train_size=0.3, random_state=42)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = to_categorical(y_train)

# define generator
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range = 0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)

# # fit generator on our train features
# datagen.fit(X_train)
# datagen.fit(X_test)

model_alex = alex()

callback = EarlyStopping(monitor='loss', patience=5, mode='min')
history = model_alex.fit(X_train, y_train, batch_size=1, epochs = 50, verbose = 1, callbacks=[callback])

model_alex.summary()

model_alex.save(filepath='modelalexnet.hdf5')

fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(121)
ax.plot(history.history["loss"])
ax.set_title("validation loss")
ax.set_xlabel("epochs")

ax2 = fig.add_subplot(122)
ax2.plot(history.history["accuracy"])
ax2.set_title("validation accuracy")
ax2.set_xlabel("epochs")
ax2.set_ylim(0, 1)

plt.show()

def carrega_data(listaimgs, imagens):
  i = 0
  amostras = []
  classe = []
  for img in listaimgs:
    imag = adaptiveThresh(imagens[i])
    amostras.append(imag)
    clas = img.split('_')
    classe.append(clas[1][0])
    i = i + 1
  return amostras, classe

amostras, classe = carrega_data(imagens, img)

X_train, X_test, y_train, y_test = train_test_split(amostras, classe, train_size=0.3, random_state=42)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = to_categorical(y_train)

# define generator
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range = 0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)

# # fit generator on our train features
# datagen.fit(X_train)
# datagen.fit(X_test)

model_le = le()

callback = EarlyStopping(monitor='loss', patience=5, mode='min')
history = model_le.fit(X_train, y_train, batch_size=1, epochs = 50, verbose = 1, callbacks=[callback])

model_le.summary()

model_le.save(filepath='modellenet.hdf5')

fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(121)
ax.plot(history.history["loss"])
ax.set_title("validation loss")
ax.set_xlabel("epochs")

ax2 = fig.add_subplot(122)
ax2.plot(history.history["accuracy"])
ax2.set_title("validation accuracy")
ax2.set_xlabel("epochs")
ax2.set_ylim(0, 1)

plt.show()