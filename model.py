# 1. Need to load the huffman tree
# 2. Build the final layer as a tanh function for turning
# left or right over the tree
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from rouge import Rouge
from bleu.bleu import Bleu
from collections import Counter
from sklearn.model_selection import KFold
from keras import backend as K
from keras.models import Model
from keras.layers import Input, LSTM, Dense, multiply, add, Concatenate, Flatten
from keras.layers import RepeatVector, Activation, Permute, Lambda, Bidirectional
from keras.layers.embeddings import Embedding
from keras.models import load_model

class SummarizationModel(object):
        def __init__(self, FLAGS, vocabulary, htree):
                super(SummarizationModel, self).__init__()
                self.vocabulary = vocabulary
                self.htree = htree
                self.FLAGS = FLAGS
                self.embed_filename = self.FLAGS.embedding_method+".6B."+\
                        str(self.FLAGS.embedding_size)+"d.txt"
                self.model_name_save = os.path.join(self.FLAGS.saved_models,
                        self.FLAGS.dataset_name+"-"+
                        self.FLAGS.embedding_method+"-"+
                        str(self.FLAGS.embedding_size))

                if self.FLAGS.multi_concat:
                        self.model_name_save += "_multi"
                else:
                        self.model_name_save += "_concat"

        def _build_graph(self):
                embedding_matrix = self._load_embed()

                encoder_inputs = Input(shape=(self.FLAGS.sentence_len,),
                        name='Encoder_input')

                enc_e = Embedding(self.FLAGS.vocab_len, self.FLAGS.embedding_size,
                        weights=[embedding_matrix], trainable=False,
                        name='EncoderEmbedding')(encoder_inputs)

                decoder_inputs = Input(shape=(self.FLAGS.timesteps,),
                        name='Decoder_input')

                dec_e = Embedding(self.FLAGS.vocab_len, self.FLAGS.embedding_size,
                        weights=[embedding_matrix], trainable=False,
                        name='DecoderEmbedding')(decoder_inputs)

                encoder_outputs, f_h, f_c, b_h, b_c = Bidirectional(
                        LSTM(self.FLAGS.units, return_sequences=True,
                        return_state=True))(enc_e)

                encoder_outputs = Dense(self.FLAGS.units)(encoder_outputs)

                state_h = Concatenate()([f_h, b_h])
                state_c = Concatenate()([f_c, b_c])
                encoder_states = [state_h, state_c]
                
                decoder_outputs = LSTM(2*self.FLAGS.units, return_sequences=True,
                        name='Decoder')(dec_e, initial_state=encoder_states)

                decoder_state = decoder_outputs = Dense(self.FLAGS.units)(decoder_outputs)

                # units == embed_dim
                # summary_len == timesteps
                # encoder_outputs : (batch_size, sentence_len, units)
                # dec_e           : (batch_size, timesteps, embed_dim)
                # decoder_outputs : (batch_size, timesteps, units)

                h = Permute((2, 1))(encoder_outputs)
                h = Dense(self.FLAGS.timesteps)(h)
                h = Permute((2, 1))(h) # (batch_size, timesteps, units)

                s = add([h, decoder_outputs])
                tanh = Activation('tanh')(s)

                tanh = Permute((2, 1))(tanh)
                tanh = Dense(self.FLAGS.sentence_len)(tanh)
                tanh = Permute((2, 1))(tanh) # (batch_size, sentence_len, units)
         
                a = Activation('softmax')(tanh)

                m = multiply([a, encoder_outputs])

                def context(m):
                        context_vector = K.sum(m, axis=1) # (batch_size, units)
                        context_vector = RepeatVector(self.FLAGS.timesteps)(context_vector)
                        return context_vector # (batch_size, timesteps, units)

                def expand_dimension(x):
                        return K.expand_dims(x, axis=-1)

                layer = Lambda(context)
                expand = Lambda(expand_dimension)

                context_vector = layer(m)

                # (batch_size, timesteps, units)
                if self.FLAGS.multi_concat:
                        decoder_outputs = multiply([context_vector, decoder_outputs])
                else:
                        decoder_outputs = Concatenate()([context_vector, decoder_outputs])

                # (batch_size, timesteps, max_depth)
                decoder_outputs = Dense(self.FLAGS.max_depth)(decoder_outputs)
                
                # (batch_size, timesteps, max_depth, 1)
                decoder_outputs = expand(decoder_outputs)

                # (batch_size, timesteps, max_depth, 3)
                p_vocab = Dense(3, activation='softmax', name='Dense')(decoder_outputs)

                return Model([encoder_inputs, decoder_inputs], p_vocab)
                
        def _load_embed(self):
                path = os.path.join(self.FLAGS.embedding_method, self.embed_filename)
                idx2word = self.vocabulary
                vocab_size = len(idx2word)
                embeddings_index = dict()

                with open(path, 'r') as f:
                        for line in f:
                                values = line.split()
                                word = values[0]
                                coefs = np.asarray(values[1:], dtype='float32')
                                embeddings_index[word] = coefs

                embedding_matrix = np.zeros((vocab_size, self.FLAGS.embedding_size))

                for i, w in idx2word.items():
                        embedding_vector = embeddings_index.get(w)
                        if embedding_vector is not None:
                                embedding_matrix[i] = embedding_vector

                return embedding_matrix   
         def train(self, train_x, train_y, val_x, val_y):

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

                model = self._build_graph()

                model.summary()

                model.compile(optimizer='rmsprop',
                        loss='categorical_crossentropy',
                        metrics=["accuracy"])

                # See train_on_batch()
                losses, loss = [], []

                for epoch in range(self.FLAGS.num_epochs):
                        print("Epoch {} of {}".format(epoch+1, self.FLAGS.num_epochs))

                        i = 0
                        while i < self.FLAGS.num_samples:
                                start = i
                                end = start + self.FLAGS.batch_size
                                batch_size = self.FLAGS.batch_size
                                
                                if end > self.FLAGS.num_samples:
                                        end = self.FLAGS.num_samples
                                        batch_size = end - start
                                        
                                encoder_input = np.array(train_x[start:end])
                                decoder_input = np.array(train_y[start:end])

                                """
                                TO USE SIMPLE SOFTMAX
                                decoder_target = \
                                        np.zeros((batch_size, self.FLAGS.timesteps,
                                                self.FLAGS.vocab_len))

                                for h, headline in enumerate(decoder_input):
                                        for t, token in enumerate(headline):
                                                if t > 0:
                                                        decoder_target[h, t-1, token] = 1"""

                                """ TO USE HIERARCHICAL CLASSIFICATION? """
                                decoder_target = \
                                        np.zeros((batch_size, self.FLAGS.timesteps,
                                                self.FLAGS.max_depth, 3))

                                decoder_target[:, :, :, -1] = 1

                                for h, headline in enumerate(decoder_input):
                                        for t, token in enumerate(headline):
                                                if t > 0:
                                                        decoder_target[h, t-1] = \
                                                                self.htree[unicode(self.vocabulary[token],
                                                                        errors='ignore')]

                                loss = model.train_on_batch([encoder_input, decoder_input],
                                        decoder_target)

                                print("\tbatch step {}/{}  ".format(
                                        end, self.FLAGS.num_samples), end="")
                                print("loss: {}, acc: {}".format(
                                        loss[0], loss[1]))
                                        
                                i += self.FLAGS.batch_size
                        losses.append(loss)

                        # Build the targets
                        """decoder_target_eval = \
                                np.zeros((val_x.shape[0], self.FLAGS.timesteps,
                                self.FLAGS.vocab_len))

                        for h, headline in enumerate(val_y):
                                for t, token in enumerate(headline):
                                        if t > 0:
                                                decoder_target_eval[h, t-1, token] = 1"""

                        decoder_target_eval = np.zeros((val_x.shape[0], self.FLAGS.timesteps,
                                self.FLAGS.max_depth, 3))

                        decoder_target_eval[:, :, :, -1] = 1

                        for h, headline in enumerate(val_y):
                                for t, token in enumerate(headline):
                                        if t > 0:
                                                decoder_target_eval[h, t-1] = \
                                                        self.htree[unicode(self.vocabulary[token], errors='ignore')]


                        # Test
                        eval_loss = model.test_on_batch(
                                [val_x, val_y], decoder_target_eval)

                        print("\tval loss: {}, acc: {}".format(
                                eval_loss[0], eval_loss[1]))

                model.save_weights(self.model_name_save+'2.h5')

                plt.plot([loss[0] for loss in losses])
                plt.title('Model loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend(['Train'], loc='upper left')
                plt.savefig(self.model_name_save+'2.png')

        def eval(self, test_x, test_y):
                def eval_rouge(ref, hyp):
                        return Rouge().get_scores(hyp, ref)

                def eval_bleu(ref, hyp):
                        return Bleu(4).compute_score({1: [ref]}, {1: [hyp]})

                def where_min(l):
                        w, v = 0, 1
                        for i, _ in enumerate(l):
                                if _ < v:
                                        w, v = i, _
                        return w

                model = self._build_graph()
                model.load_weights(self.model_name_save+'.h5')

                rouge_metrics={'rouge-l': 0.0, 'rouge-2': 0.0, 'rouge-1': 0.0}
                bleu_metrics={'bleu-4': 0.0, 'bleu-3': 0.0, 'bleu-2': 0.0, 'bleu-1': 0.0}

                hypf = open(
                        os.path.join(self.FLAGS.hypothesis_references,
                                self.FLAGS.dataset_name+"_"+
                                self.FLAGS.embedding_method+
                                str(self.FLAGS.embedding_size)+".hyp"),
                        "w+")
                reff = open(
                        os.path.join(self.FLAGS.hypothesis_references,
                                self.FLAGS.dataset_name+"_"+
                                self.FLAGS.embedding_method+
                                str(self.FLAGS.embedding_size)+".ref"),
                        "w+")

                i, dec = 0, 0
                num_samples = test_x.shape[0]

                while i < num_samples:
                        start = i
                        end = start + self.FLAGS.batch_size
                        batch_size = self.FLAGS.batch_size
                        if end > num_samples:
                                end = num_samples
                                batch_size = end - start

                        batch_x, batch_y = test_x[start:end], test_y[start:end]

                        predicted = model.predict_on_batch([batch_x, batch_y])

                        for y, pred in zip(batch_y, predicted):
                                ytrue = [self.vocabulary[int(t)] for t in y if t > 0]
                                ypred = []
                                for token in pred:
                                        likelihoods = []
                                        for k,v in self.htree.items():
                                                cdist = cosine(np.array(token).flatten(),
                                                        np.array(v).flatten())
                                                likelihoods.append([k, cdist])
                                        j = where_min([l[-1] for l in likelihoods])
                                        ypred.append(likelihoods[j][0])

                                ytrue = ' '.join([token for token in ytrue
                                        if token not in ["<go>", "<pad>", "<eos>"]])
                                ypred = ' '.join([token for token in ypred
                                        if token not in ["<go>", "<pad>", "<eos>"]])

                                try:
                                        rouge = eval_rouge(ytrue, ypred)
                                        bleu, _ = eval_bleu(ytrue, ypred)

                                        hypf.write('{}\n'.format(ypred))
                                        reff.write('{}\n'.format(ytrue))

                                        print("reference: {}".format(ytrue))
                                        print("hypothesis: {}".format(ypred))

                                        for key in rouge_metrics.keys():
                                                rouge_metrics[key] += rouge[0][key]['f']
                                        for index, key in enumerate(bleu_metrics.keys()):
                                                bleu_metrics[key] += bleu[index]

                                except Exception as e:
                                        dec += 1
                                        print(e)
                                        next

                        i += self.FLAGS.batch_size

                print('given {} samples'.format(num_samples-dec))
                print([(k, v/(num_samples-dec)) for k, v in rouge_metrics.iteritems()])
                print([(k, v/(num_samples-dec)) for k, v in bleu_metrics.iteritems()])
