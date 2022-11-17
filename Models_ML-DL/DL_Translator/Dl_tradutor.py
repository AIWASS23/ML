import numpy as np # linear algebra
import pandas as pd # data processing
import os
import string
import matplotlib.pyplot as plt
import re
import seaborn as sns

from string import digits
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model

lines = pd.read_csv("Hindi_English_Truncated_Corpus.csv",encoding = 'utf-8')
lines = lines[lines['source'] == 'ted']
lines = lines[~pd.isnull(lines['english_sentence'])]
lines.drop_duplicates(inplace = True)

# Escolhendo quaisquer 25.000 linhas do conjunto de dados

lines = lines.sample(n = 25000, random_state = 42)
lines.shape

# colocar todos os caracteres em minúsculas no conjunto de dados

lines['english_sentence'] = lines['english_sentence'].apply(lambda x: x.lower())
lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: x.lower())

# remover todas as aspas dos dados

lines['english_sentence'] = lines['english_sentence'].apply(lambda x: re.sub("'", '', x))
lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: re.sub("'", '', x))

exclude = set(string.punctuation) # Conjunto de todos os caracteres especiais

# Remova todos os caracteres especiais

lines['english_sentence'] = lines['english_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

# remover todos os números e espaços extras dos dados

remove_digits = str.maketrans('', '', digits)
lines['english_sentence']=lines['english_sentence'].apply(lambda x: x.translate(remove_digits))
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: x.translate(remove_digits))

lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: re.sub("[२३०८१५७९४६]", "", x))

# Remover espaços extras

lines['english_sentence']=lines['english_sentence'].apply(lambda x: x.strip())
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: x.strip())
lines['english_sentence']=lines['english_sentence'].apply(lambda x: re.sub(" +", " ", x))
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: re.sub(" +", " ", x))

lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x : 'START_ '+ x + ' _END')

# Obtenha vocabulário em inglês e hindi

all_eng_words = set()
for eng in lines['english_sentence']:
    for word in eng.split():
        if word not in all_eng_words:
            all_eng_words.add(word)

all_hindi_words=set()
for hin in lines['hindi_sentence']:
    for word in hin.split():
        if word not in all_hindi_words:
            all_hindi_words.add(word)
lines['length_eng_sentence']=lines['english_sentence'].apply(lambda x:len(x.split(" ")))
lines['length_hin_sentence']=lines['hindi_sentence'].apply(lambda x:len(x.split(" ")))

# modelo de treino

X, y = lines['english_sentence'], lines['hindi_sentence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 42)

X_train.to_pickle('X_train.pkl')
X_test.to_pickle('X_test.pkl')

# treinar o modelo de tradução de idiomas

def generate_batch(X = X_train, y = y_train, batch_size = 128):
    ''' Gerar dados '''
    while True:
        for j in range(0, len(X), batch_size):
            encoder_input_data = np.zeros((batch_size, max_length_src),dtype = 'float32')
            decoder_input_data = np.zeros((batch_size, max_length_tar),dtype = 'float32')
            decoder_target_data = np.zeros((batch_size, max_length_tar, num_decoder_tokens),dtype = 'float32')

            for i, (input_text, target_text) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                for t, word in enumerate(input_text.split()):
                    encoder_input_data[i, t] = input_token_index[word] # sequência de entrada do codificador

                for t, word in enumerate(target_text.split()):
                    if t < len(target_text.split())-1:
                        decoder_input_data[i, t] = target_token_index[word] # sequência de entrada do decodificador

                    if t > 0:
                        # Sequência de destino do decodificador
                        # não inclui o START_token 
                        # Deslocamento por um passo de tempo'''
                        decoder_target_data[i, t - 1, target_token_index[word]] = 1.
            yield([encoder_input_data, decoder_input_data], decoder_target_data)
          
latent_dim = 300
encoder_inputs = Input(shape=(None,))
enc_emb =  Embedding(num_encoder_tokens, latent_dim, mask_zero = True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

# Descartar o `encoder_outputs` e manter apenas os estados

encoder_states = [state_h, state_c]
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero = True)
dec_emb = dec_emb_layer(decoder_inputs)

# Configura o decodificador para retornar sequências de saída completas, e também para retornar estados internos. Retorna estados no modelo de treinamento usados na inferência.'''

decoder_lstm = LSTM(latent_dim, return_sequences = True, return_state = True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state = encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation = 'softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Defina o modelo que vai virar
# encoder_input_data` & `decoder_input_data` em `decoder_target_data`

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.summary()

train_samples = len(X_train)
val_samples = len(X_test)
batch_size = 128
epochs = 100

model.fit_generator(
    generator = generate_batch(X_train, y_train, batch_size = batch_size),
    steps_per_epoch = train_samples//batch_size,
    epochs = epochs,
    validation_data = generate_batch(X_test, y_test, batch_size = batch_size),
    validation_steps = val_samples//batch_size)

model.save_weights('nmt_weights.h5')

# Codifique a sequência de entrada para obter os vetores

encoder_model = Model(encoder_inputs, encoder_states)

# Configuração do decodificador. Os tensores abaixo manterão os estados da etapa de tempo anterior

decoder_state_input_h = Input(shape = (latent_dim,))
decoder_state_input_c = Input(shape = (latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = dec_emb_layer(decoder_inputs) # Obtenha os embeddings da sequência do decodificador

# Para prever a próxima palavra na sequência, defina os estados iniciais para os estados do intervalo de tempo anterior

decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2) # Uma dense softmax layer para gerar prob dist. Sobre o vocabulário alvo

# Modelo final do decodificador

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)
    
def decode_sequence(input_seq):
    # Codifique a entrada como vetores de estado.
    states_value = encoder_model.predict(input_seq)
    # Gera uma sequência alvo vazia de comprimento 1.
    target_seq = np.zeros((1,1))
    # Preencha o primeiro caractere da sequência de destino com o caractere inicial.
    target_seq[0, 0] = target_token_index['START_']

    # Loop de amostragem para umas sequências para simplificar.
    
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # amostra de token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' '+sampled_char

        # Condição de saída: atingir o comprimento máximo ou encontre o caractere de parada.
        if (sampled_char == '_END' or
           len(decoded_sentence) > 50):
            stop_condition = True

        # Atualize a sequência de destino (de comprimento 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Atualizar estados
        states_value = [h, c]

    return decoded_sentence

train_gen = generate_batch(X_train, y_train, batch_size = 1)
k=-1

