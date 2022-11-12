
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Flatten,Dense
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,TensorBoard
from PDI.src.pdi_utils import load_sign_language_data


#carregando informações do modelo
data_train , label_train , data_val , label_val, _ , _  = load_sign_language_data()

#inicialização do modelo sequencial do keras
model =keras.Sequential()

#adição de camada convolucional 2D
model.add(Conv2D(16, kernel_size=(5, 5), activation= 'relu',input_shape=data_train.shape[1:],padding='same'))

#adição da camada de redução maxpooling
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

#transoformação dos dados em 1 dimensão
model.add(Flatten())

#adição da camada densa
model.add(Dense(29, activation = 'softmax'))

#compilação do modelo
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['sparse_categorical_accuracy'])


#definição do nome do modelo
model_name = 'PDI/Aula_4/aula4_code4_models'

#definição do arquivo salvo do modelo
model_file = model_name + '.h5'

#callback para usar tensorboard
tensorboard = TensorBoard(log_dir="logs/{}".format(model_file))

#callback para salvar o modelo
mcp_save= ModelCheckpoint(model_file, save_best_only = True, monitor = 'val_loss', mode = 'min')

#callback para interromper o treinamento quando não houver mais aprendizado
earlyStopping= EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1, mode = 'min')

#callback para reduzira a taxa de aprendizagem do modelo
reduce_lr_loss = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.05, patience = 5, verbose = 1, min_delta = 1e-4, mode = 'min')

#treinamento do modelo
model.fit(data_train, label_train, validation_data = (data_val, label_val), epochs = 50, callbacks = [mcp_save, earlyStopping, reduce_lr_loss, tensorboard])