"""
NOTES:
        model_multi -> model which the context and the decoder outputs are multiplied
        model_concat -> same as above, but concatenated instead of multiplied

TODO:
        1) implement this on tensorflow
        2) load data in batches
"""

from __future__ import print_function
from keras import backend as K
from sklearn.cross_validation import train_test_split
from keras.models import Model
from keras.layers import Input, LSTM, Dense, multiply, add, Concatenate
from keras.layers import RepeatVector, Activation, Permute, Lambda, Bidirectional
from keras.layers.embeddings import Embedding
from keras.models import load_model, model_from_json
from rouge import Rouge
from bleu.bleu import Bleu
from keras.utils import plot_model
import cPickle as pickle
import numpy as np
import random
import tensorflow as tf
import matplotlib as mpl
import os
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

batch_size = 32
num_epochs = 15
embedding_size = 100
units = 128
filename_model = 'saved_models/model_multi'
data = 'bbc_news'
model_save_name = filename_model+'-'+str(embedding_size)

def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label
        classification of how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives/(predicted_positives + K.epsilon())
        return precision

def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_falses = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (predicted_falses + K.epsilon())
        return recall

def build_sentence(y_true, y_pred):
        ref = ""
        hyp = ""
        for token in y_true:
                if token not in ["<go>", "<pad>", "<eos>"]:
                        ref += token+" "
        for token in y_pred:
                if token not in ["<go>", "<pad>", "<eos>"]:
                        hyp += token+" "
        return hyp, ref

def eval_rouge(ref, hyp):
        return Rouge().get_scores(hyp, ref)

def eval_bleu(ref, hyp):
        return Bleu(4).compute_score({1: [ref]}, {1: [hyp]})
def load_data():
        with open('vocabs/'+data+'.vocab.data.pkl', 'rb') as fp:
                X, Y = pickle.load(fp)

        with open('vocabs/'+data+'.vocab.pkl', 'rb') as fp:
                idx2word, word2idx = pickle.load(fp)

        return X, Y, idx2word, word2idx

def padding(sentences):
        padding_len = max(len(sentence) for sentence in sentences)
        for sentence in sentences:
                sentence += [0 for _ in range(padding_len-len(sentence))]

def load_embed(path, idx2word):
        vocab_size = len(idx2word)
        embeddings_index = dict()

        with open(path, 'r') as f:
                for line in f:
                        values = line.split()
                        word = values[0]
                        coefs = np.asarray(values[1:], dtype='float32')
                        embeddings_index[word] = coefs

        embedding_matrix = np.zeros((vocab_size, embedding_size))

        for i, w in idx2word.items():
                embedding_vector = embeddings_index.get(w)
                if embedding_vector is not None:
                        embedding_matrix[i] = embedding_vector
        return embedding_matrix

def create_model_pipeline(num_tokens):
        """
        Cria a arquitetura da rede recorrente.
        Eh necessario chamar essa funcao sempre para que seja possivel carregar...
        ...o modelo treinado para que as predicoes sejam possiveis.
        Por enquanto, ha somente a criacao abstrata da rede e os dados nao fluem.
        """
        if embedding_size == 100:
                embedding_matrix = \
                        load_embed('glove/glove.6B.100d.txt', idx2word)
        elif embedding_size == 300:
                embedding_matrix = \
                        load_embed('glove/glove.6B.300d.txt', idx2word)

        encoder_inputs = Input(shape=(sentence_len,),
                name='Encoder_input')
        enc_e = Embedding(num_tokens, embedding_size,
                weights=[embedding_matrix], trainable=False,
                name='EncoderEmbedding')(encoder_inputs)

        decoder_inputs = Input(shape=(timesteps,),
                name='Decoder_input')

        dec_e = Embedding(num_tokens, embedding_size,
                weights=[embedding_matrix], trainable=False,
                name='DecoderEmbedding')(decoder_inputs)

        encoder_outputs, f_h, f_c, b_h, b_c = Bidirectional(LSTM(units,
                return_sequences=True, return_state=True))(enc_e)

        encoder_outputs = Dense(units)(encoder_outputs)

        state_h = Concatenate()([f_h, b_h])
        state_c = Concatenate()([f_c, b_c])
        encoder_states = [state_h, state_c]

        decoder_outputs = LSTM(2*units, return_sequences=True,
                name='Decoder')(dec_e, initial_state=encoder_states)

        decoder_outputs = Dense(units)(decoder_outputs)
        # units == embed_dim
        # summary_len == timesteps
        # encoder_outputs : (batch_size, sentence_len, units)
        # dec_e           : (batch_size, timesteps, embed_dim)
        # decoder_outputs : (batch_size, timesteps, units)

        h = Permute((2, 1))(encoder_outputs)
        h = Dense(timesteps)(h)
        h = Permute((2, 1))(h) # (batch_size, timesteps, units)

        s = add([h, decoder_outputs])
        tanh = Activation('tanh')(s)

        tanh = Permute((2, 1))(tanh)
        tanh = Dense(sentence_len)(tanh)
        tanh = Permute((2, 1))(tanh) # (batch_size, sentence_len, units)

        a = Activation('softmax')(tanh)

        m = multiply([a, encoder_outputs])

        def context(m):
                context_vector = K.sum(m, axis=1) # (batch_size, units)
                context_vector = RepeatVector(timesteps)(context_vector)
                return context_vector # (batch_size, timesteps, units)

        layer = Lambda(context)

        context_vector = layer(m)
        decoder_outputs = multiply([context_vector, decoder_outputs]) # use concat(?), instead
        #decoder_outputs = Concatenate()([context_vector, decoder_outputs])

        decoder_outputs = Dense(num_tokens,
                activation='softmax', name='Dense')(decoder_outputs)
        return encoder_inputs, decoder_inputs, decoder_outputs

def create_model():
        """
        Cria o objeto do modelo.
        Inputs:
                encoder_inputs: Tensor do documento a ser sumarizado
                decoder_inputs: Tensor do resumo de referencia
        Outputs:
                decoder_outputs: Tensor do resumo gerado pela rede recorrente
        """
        encoder_inputs, decoder_inputs, decoder_outputs = \
                create_model_pipeline(num_tokens)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.summary()

        model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=["accuracy", precision, recall])

        return model

def train(model):
        """
        Treina o modelo
        Inputs:
                encoder_input: Documento a ser sumarizado
                decoder_input: Resumo de referencia
        Outputs:
                decoder_target: Resumo gerado
        """
        history = model.fit([encoder_input, decoder_input], decoder_target,
                batch_size=batch_size,
                epochs=num_epochs,
                validation_split=0.1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(model_save_name+'.png')

        model_json = model.to_json()
        with open(model_save_name+'.json', 'w') as json_file:
                json_file.write(model_json)
        model.save_weights(model_save_name+'.h5')

def evaluate_model(model):
        with open(filename_model+'-'+str(embedding_size)+'.json', 'r') as json_file:
                loaded_model_json = json_file.read()

        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(filename_model+'-'+str(embedding_size)+'.h5')

        loaded_model.compile(optimizer='adam',
                loss='categorical_crossentropy', metrics=[precision, recall])

        rouge_metrics={'rouge-l': 0.0, 'rouge-2': 0.0, 'rouge-1': 0.0}
        bleu_metrics={'bleu-4': 0.0, 'bleu-3': 0.0, 'bleu-2': 0.0, 'bleu-1': 0.0}

        #randrange = np.random.randint(0, encoder_input.shape[0], size=2876)
        documents = encoder_eval
        summaries = decoder_eval

        descont_samples = 0

        hypf = open(model_save_name+'_hypotesis_eval.txt', 'w+')
        reff = open(model_save_name+'_references_eval.txt', 'w+')
        
        for document, summary in zip(documents, summaries):
                try:
                        x = np.array([t for t in document])
                        x = np.reshape(x, (1, x.shape[0]))
                        y = np.reshape(summary, (1, summary.shape[0]))

                        predicted = loaded_model.predict([x, y])

                        ytrue = [idx2word[int(t)] for t in y[0] if t > 0]
                        ypred = [idx2word[np.argmax(t)] for t in predicted[0]]

                        hyp, ref = build_sentence(ytrue, ypred)

                        hypf.write('{}\n'.format(hyp))
                        reff.write('{}\n'.format(ref))

                        rouge_scores = eval_rouge(ref, hyp)
                        bleu_scores, _ = eval_bleu(ref, hyp)

                        for key in rouge_metrics.keys():
                                rouge_metrics[key] += rouge_scores[0][key]['f']

                        for index, key in enumerate(bleu_metrics.keys()):
                                bleu_metrics[key] += bleu_scores[index]
                except UnicodeDecodeError:
                        descont_samples += 1
                        pass

        num_samples = summaries.shape[0] - descont_samples
        print('given {} samples'.format(num_samples))
        print([(k, v/num_samples) for k, v in rouge_metrics.iteritems()])
        print([(k, v/num_samples) for k, v in bleu_metrics.iteritems()])

X, Y, idx2word, word2idx = load_data()

vocab_len = len(word2idx)
padding(X)
padding(Y)
X = np.array([np.array(sentences) for sentences in X])
Y = np.array([np.array(sentences) for sentences in Y])

X_train, X_eval, Y_train, Y_eval = \
        train_test_split(X, Y, test_size=0.25, random_state=42)

num_samples = X_train.shape[0]
sentence_len = X.shape[-1]
timesteps = Y.shape[-1]

# These are used for training
encoder_input = np.zeros((num_samples, sentence_len),
        dtype='float32')
decoder_input = np.zeros((num_samples, timesteps),
        dtype='float32')
target = np.zeros((timesteps, vocab_len),
        dtype='float32')
decoder_target = np.zeros((num_samples,))
decoder_target = np.zeros((num_samples, timesteps, vocab_len),
        dtype='float32')

# These are used for evaluation
encoder_eval = np.zeros((X_eval.shape[0], sentence_len),
     dtype='float32')
decoder_eval = np.zeros((Y_eval.shape[0], timesteps),
     dtype='float32')

# Feed with real values
for i, (descr, headline) in enumerate(zip(X_train, Y_train)):
        for t, token_idx in enumerate(descr):
                encoder_input[i, t] = token_idx
        for t, token_idx in enumerate(headline):
                decoder_input[i, t] = token_idx
                if t>0:
                        decoder_target[i, t-1, token_idx] = 1

for i, (descr, headline) in enumerate(zip(X_eval, Y_eval)):
        for t, token_idx in enumerate(descr):
                encoder_eval[i, t] = token_idx
        for t, token_idx in enumerate(headline):
                decoder_eval[i, t] = token_idx

num_tokens = len(word2idx)


print('news headlines format:', Y_train.shape)
print('news descriptions format:', X_train.shape)
print('number of tokens in the vocabulary', len(word2idx))


# Model architecture
model = create_model()

# Training the model
train(model)

# Evaluation starts here
#evaluate_model(model)
#print(model_save_name)
