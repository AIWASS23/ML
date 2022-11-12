
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Flatten,Dense
from PDI.src.pdi_utils import load_sign_language_data
import sys
sys.path.append(' . ')
import os

#função para salvar sumario
def save_sumary(name):

    with open('sumary/'+name +'.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

#função para definir callbacks
def callbacks(model_name):

    tensorboard = TensorBoard(log_dir="logs/{}".format(model_name))
    mcp_save = ModelCheckpoint(model_file ,save_best_only=True, monitor='val_loss', mode='min')
    earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=2, verbose=1,min_delta=1e-4,mode='min')
    return tensorboard,mcp_save,earlyStopping,reduce_lr_loss


#carregando informaçoes do dataset
data_train , label_train , data_val , label_val, _ , _  = load_sign_language_data()

#definição da quantidade de filtros em cada camada convolucional
conv_node=[8,16,32,64]

#definição da quantidade de neuronios da ultima camada
output = 29

#definição da quantidade de filtros da primeira camada
first_node = 8

#iteração para determinar as variações de cnn
for cnn in  range(len(conv_node)):

    #inicialização do modelo sequencial do keras
    model = keras.Sequential()

    #adição da primeira camada convolucional
    model.add(Conv2D(first_node, kernel_size= (3,3), activation = 'relu', padding = 'same'), input_shape = data_train.shape[1:])

    #adição da camada de redução maxpooling
    model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))

    #adição da camada dropout para eliminar 25% de dados aleatórios do treinamento
    model.add(Dropout(.25))

    #iteração para criar blocos adicionais de caamdas convolucionais.
    for node in conv_node[:cnn]:
        #adição de camada convolucional
        model.add(Conv2D(node, kernel_size = (3,3), activation = 'relu', padding = 'same'))

        #adição de camada de redução maxpooling
        model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))

    #transormaçao dos dados em 1 dimensão
    model.add(Flatten())

    #adição da camada de densa de saída
    model.add(Dense(29, activation = 'softmax'))

    #compilação do modelo
    model.compile(
        loss = 'sparse_categorial_crossentropy',
        optimizer = 'adam',
        metrics = ['sparse_categorial_accuracy'])

    #definição do nome do modelo salvo em arquivo
    model_file = 'PDI/Aula_4/aula4_code5_models/{}-block-{}-convNode.h5'.format(cnn,conv_node[:cnn])

    #nome do log que será analisado pelo tensorboard
    model_name = "{}-block-{}-convNOde".format(cnn,conv_node[:cnn])

    #definição dos callbacks
    tensorboard,mcp_save,earlyStopping,reduce_lr_loss = callbacks(model_name)

    #salvando sumario dos modelos
    save_sumary(model_name)

    #treinando modelo
    model.fit(
        data_train,
        label_train,
        validation_data = (data_val, label_val),
        epochs = 5,
        callbacks = [
            tensorboard,
            mcp_save,
            earlyStopping,
            reduce_lr_loss])
