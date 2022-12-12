
# Commented out IPython magic to ensure Python compatibility.
import os, random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import classification_report
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.optimizers import Nadam, Adagrad, Adadelta
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Masking, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils, to_categorical, plot_model 
from keras.applications import VGG16, Xception, InceptionV3 as Inception, DenseNet121 as DenseNet
import random
import tensorflow as tf
import cv2 as cv
import os 
import glob
import warnings 
warnings.filterwarnings('always') 
warnings.filterwarnings('ignore')
# %matplotlib inline

from google.colab import drive
drive.mount('/content/drive')

"""##Carregando o conjuntos de dados"""

fire_dir = glob.glob(os.path.join('/content/drive/MyDrive/CV/img_data/train/fire/', '*'))
smoke_dir = glob.glob(os.path.join('/content/drive/MyDrive/CV/img_data/train/smoke/', '*'))
default_dir = glob.glob(os.path.join('/content/drive/MyDrive/CV/img_data/train/default/', '*'))
X_path = fire_dir + smoke_dir + default_dir
X = []
for f in X_path:
  X.append(np.array(cv.resize(cv.imread(f), (224,224), interpolation = cv.INTER_AREA)))
X = np.array(X)
X = X / 255

l_fire = np.zeros(len(fire_dir))
l_smoke = np.ones(len(smoke_dir))
l_default = 2*np.ones(len(default_dir))
y = np.concatenate((l_fire, l_smoke, l_default))
y = to_categorical(y, 3)

#show sample images
nb_rows = 3
nb_cols = 5
dic = {0:'fire',1:'smoke',2:'default'}

fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(12, 7));
for i in range(0, nb_rows):
    for j in range(0, nb_cols):
        path = f'//content/drive/MyDrive/CV/img_data/train/{dic[i]}'
        choice = random.choice(os.listdir(path))
        axs[i, j].xaxis.set_ticklabels([]);
        axs[i, j].yaxis.set_ticklabels([]);
        axs[i, j].imshow((plt.imread(os.path.join(path,choice))));
plt.show();

"""## Separação dos dados em treino e teste"""

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=42)

print(X_train.shape)
print(X_val.shape)
print(y_val.shape)
print(y_train.shape)

"""## Data augmentation"""

datagen = ImageDataGenerator(
        zoom_range = 0.1, # Aleatory zoom
        rotation_range= 15, 
        width_shift_range=0.1,  # horizontal shift
        height_shift_range=0.1,  # vertical shift
        horizontal_flip=True,  
        vertical_flip=True)
datagen.fit(X_train)

"""## Treinando as Redes Neurais"""

inp = Input((224,224,3))
conv1 = Conv2D(64, (5,5), padding='valid', activation= 'relu')(inp)
conv1 = MaxPooling2D(pool_size=(2,2))(conv1)
conv1 = BatchNormalization()(conv1)
conv2 = Conv2D(96, (4,4), padding='valid', activation= 'relu')(conv1)
conv2 = MaxPooling2D(pool_size=(2,2))(conv2)
conv2 = BatchNormalization()(conv2)
conv3 = Conv2D(128, (3,3), padding='valid', activation= 'relu')(conv2)
conv3 = MaxPooling2D(pool_size=(2,2))(conv3)
conv3 = BatchNormalization()(conv3)
conv4 = Conv2D(256, (3,3), padding='valid', activation= 'relu')(conv3)
conv4 = MaxPooling2D(pool_size=(2,2))(conv4)
conv4 = BatchNormalization()(conv4)
flat = Flatten()(conv4)
dense1 = Dense(512, activation= 'relu')(flat)
dense1 = Dropout(0.5)(dense1)
dense2 = Dense(64, activation= 'relu')(dense1)
dense2 = Dropout(0.1)(dense2)
out = Dense(3, activation = 'softmax')(dense2)
model = Model(inp, out)
model.compile(optimizer = Adagrad(lr = 0.0001) , loss = 'categorical_crossentropy', metrics=['accuracy'])

CNN = model.fit(X_train, y_train, batch_size = 16, epochs = 50, initial_epoch = 0, validation_data = (X_val, y_val))

plot_model(model,show_layer_names=True,show_dtype=True,show_shapes=True)

CNN_pred = model.predict(X_val)
CNN_pred_list = np.argmax(CNN_pred, axis = 1)

"""## VGG"""

VGG = VGG16(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(224, 224, 3),
    pooling=None,
    classifier_activation="softmax"
)

x = VGG.output
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(256,activation='relu')(x) 
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(256,activation='relu')(x) 
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(3,activation='softmax')(x)
VGG_model = keras.Model(inputs=VGG.input, outputs=out)

for layer in VGG.layers[:20]:
    layer.trainable=False

VGG_model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])

VGG_history = VGG_model.fit(X_train, y_train, batch_size = 16, epochs = 50, initial_epoch = 0, validation_data = (X_val, y_val))

plot_model(VGG_model,show_layer_names=True,show_dtype=True,show_shapes=True)

VGG_pred = VGG_model.predict(X_val)
VGG_pred_list = np.argmax(VGG_pred, axis = 1)

"""## Xception"""

Xception = Xception(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(224, 224, 3),
    pooling=None,
    classifier_activation="softmax"
)

x = Xception.output
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(256,activation='relu')(x) 
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(256,activation='relu')(x) 
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(3,activation='softmax')(x)
Xception_model = keras.Model(inputs=Xception.input, outputs=out)

for layer in Xception.layers[:20]:
    layer.trainable=False

Xception_model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])

Xception_history = Xception_model.fit(X_train, y_train, batch_size = 16, epochs = 50, initial_epoch = 0, validation_data = (X_val, y_val))

plot_model(Xception_model,show_layer_names=True,show_dtype=True,show_shapes=True)

Xception_pred = Xception_model.predict(X_val)
Xception_pred_list = np.argmax(Xception_pred, axis = 1)

"""## INCEPTION"""

Inception = Inception(   #tf.keras.applications.inception_v3.InceptionV3 
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(224, 224, 3),
    pooling=None,
    classes=3,
    classifier_activation='softmax'
)

x = Inception.output
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(256,activation='relu')(x) 
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(256,activation='relu')(x) 
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(3,activation='softmax')(x)
Inception_model = keras.Model(inputs=Inception.input, outputs=out)

for layer in Inception.layers[:20]:
    layer.trainable=False

Inception_model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])

Inception_history = Inception_model.fit(X_train, y_train, batch_size = 16, epochs = 50, initial_epoch = 0, validation_data = (X_val, y_val))

Inception_pred = Inception_model.predict(X_val)
Inception_pred_list = np.argmax(Inception_pred, axis = 1)

"""## DENSENET"""

DenseNet = DenseNet(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(224, 224, 3),
    pooling=None,
    classes=3,
    classifier_activation='softmax'
)

x = DenseNet.output
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(256,activation='relu')(x) 
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(256,activation='relu')(x) 
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(3,activation='softmax')(x)
DenseNet_model = keras.Model(inputs=DenseNet.input, outputs=out)

for layer in DenseNet.layers[:20]:
    layer.trainable=False
DenseNet_model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])

DenseNet_history = DenseNet_model.fit(X_train, y_train, batch_size = 16, epochs = 50, initial_epoch = 0, validation_data = (X_val, y_val))

DenseNet_pred = DenseNet_model.predict(X_val)
DenseNet_pred_list = np.argmax(DenseNet_pred, axis = 1)

models = ['CNN', 
          'VGG', 
          'Xception',
          'Inception',
          'DenseNet'
          ]

results = []
results.append(CNN_pred_list)
results.append(VGG_pred_list)
results.append(Xception_pred_list)
results.append(Inception_pred_list)
results.append(DenseNet_pred_list)
results = np.asarray(results)

for i in range(len(models)):
  joblib.dump(f'{models[i]}_model', f'//content/drive/MyDrive/Pyoneers/MachineLearning/{models[i]}.joblib')

"""##Avaliando os modelos

#### As métricas utilizados para avaliar o desempenho dos modelos de classificação foram:

- **Acurácia**;
- **Recall**;
- **Precisão**;
- **F1-Score**. <br>
"""

plt.style.use("ggplot")
fig = plt.figure()
fig.set_size_inches(28, 5)

fig.add_subplot(1, 5, 1)
plt.plot(np.arange(0, 50), CNN.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), CNN.history['accuracy'], label="val_acc")
plt.title("CNN")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.ylim([0, 1.1])

fig.add_subplot(1, 5, 2)
plt.plot(np.arange(0, 50), VGG_history.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), VGG_history.history['accuracy'], label="val_acc")
plt.title("VGG")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.ylim([0, 1.1]);

fig.add_subplot(1, 5, 3)
plt.plot(np.arange(0, 50), Xception_history.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), Xception_history.history['accuracy'], label="val_acc")
plt.title("Xception")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.ylim([0, 1.1]);

fig.add_subplot(1, 5, 4)
plt.plot(np.arange(0, 50), Inception_history.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), Inception_history.history['accuracy'], label="val_acc")
plt.title("Inception")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.ylim([0, 1.1]);

fig.add_subplot(1, 5, 5)
plt.plot(np.arange(0, 50), DenseNet_history.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), DenseNet_history.history['accuracy'], label="val_acc")
plt.title("DenseNet")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.ylim([0, 1.1]);

def get_confusion_matrix(y_true, y_pred, model):
    plt.subplots(figsize=(8,6))
    labels = ['fire','smoke','default']
    cm = metrics.confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', cmap='YlGnBu').set(title=model, ylabel='Real', xlabel='Predito', 
                                                            yticklabels=labels,
                                                            xticklabels=labels)

y_val_list = np.argmax(y_val, axis = 1)

for i in range(len(models)):
  print(models[i])
  print(classification_report(y_val_list, results[i]))
  get_confusion_matrix(y_val_list, results[i], models[i])
  print('\n')

"""## Conclusão

Todas as pipelines apresentaram desempenho satisfatório (Precisão e Recall > 0,9). O melhor modelo, dentre os testados, foi o VGG.
"""