import cPickle as pickle
import numpy as np

def load_data(arg):
        with open(arg+".vocab.data.pkl", "rb") as fp:
                X, Y = pickle.load(fp)

        with open(arg+".vocab.pkl", "rb") as fp:
                idx2word, word2idx = pickle.load(fp)

        with open(arg+".tree.pkl" , "rb") as fp:
                tree = pickle.load(fp)

        return X, Y, idx2word, word2idx, tree


def padding(arg2pad):
        if type(arg2pad) is dict:
                arg2pad.update({'<pad>':'00', '<go>':'01', '<eos>':'10', '<unk>':'11'})
                max_depth = max([len(path) for path in arg2pad.values()])
                for k,v in arg2pad.iteritems():
                        arg2pad[k] = [[1, 0, 0] if int(c) == 0 else [0, 1, 0] for c in v]
                        if len(arg2pad[k]) < max_depth:
                                arg2pad[k] += \
                                        [[0, 0, 1] for _ in range(max_depth - len(arg2pad[k]))]
        else:
                padding_len = max(len(sentence) for sentence in arg2pad)
                for sentence in arg2pad:
                        sentence += [0 for _ in range(padding_len-len(sentence))]

def get_data(datapath, test_size, val_size):
        X, Y, idx2word, word2idx, htree = load_data(datapath)

        padding(X)
        padding(Y)

        assert len(X) == len(Y), "Wrong dimension comparison"
        padding(htree)

        testing_size = int(len(X)*test_size)

        train_x = np.array(X[:-testing_size])
        train_y = np.array(list(Y[:-testing_size]))
        test_x = np.array(list(X[-testing_size:]))
        test_y = np.array(list(Y[-testing_size:]))

        validation_size = int(train_x.shape[0] * val_size)

        val_x = np.array(list(train_x[-validation_size:]))
        val_y = np.array(list(train_y[-validation_size:]))
        train_x = np.array(list(train_x[:-validation_size]))
        train_y = np.array(list(train_y[:-validation_size]))

        print("training set: ", train_x.shape, train_y.shape)
        print("validation set: ", val_x.shape, val_y.shape)
        print("test set: ", test_x.shape, test_y.shape)

        return [train_x, train_y, val_x, val_y, test_x, test_y], [idx2word, word2idx], htree
