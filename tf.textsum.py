from __future__ import print_function
from model import SummarizationModel
from utils import get_data
import argparse
import os
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="Train the model")
parser.add_argument("--eval", help="Evaluate the model")
parser.add_argument("--load_embed",
        default="glove", help="Give a pre-trained embedding to be used")
parser.add_argument("--attlayer", default="multi",
        help="Choose between multiply or concatenate in attention layer")
parser.add_argument("--dim_embed", default=100,
        help="Set the embedding dimension")

tf.app.flags.DEFINE_integer("vocab_len", 0,
        "Vocabulary length. Alias of num_tokens")
tf.app.flags.DEFINE_integer("max_depth", 0,
        "Maximum depth of the Huffmann Tree")
tf.app.flags.DEFINE_integer("num_samples", 0, "Number of samples (documents)")
tf.app.flags.DEFINE_integer("sentence_len", 0, "Max sentence length")
tf.app.flags.DEFINE_integer("timesteps", 0, "Timesteps for a summary")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size")
tf.app.flags.DEFINE_integer("num_epochs", 1, "Epochs to train the model")
tf.app.flags.DEFINE_integer("embedding_size", 0, "Embedding size")
tf.app.flags.DEFINE_integer("units", 128, "Units for the LSTMs")
tf.app.flags.DEFINE_integer("seed", 7, "Seed for sklearn")

tf.app.flags.DEFINE_float("test_size", 0.25,
        "Dataset percentage to be used as test data")

tf.app.flags.DEFINE_float("val_size", 0.10,
        "Dataset percentage to be used as validation data")

tf.app.flags.DEFINE_string("dataset_name", "",
        "The name of the dataset to work with")
tf.app.flags.DEFINE_string("vocabs", "vocabs",
        "Path to the dataset json files")
tf.app.flags.DEFINE_string("saved_models", "saved_models",
        "Path to the saved files of the trained model")
tf.app.flags.DEFINE_string("hypothesis_references", "hypothesis+references",
        "Path to the hypothesis and references summaries")
tf.app.flags.DEFINE_string("embedding_method", "",
        "A pre-trained embedding method")

tf.app.flags.DEFINE_boolean("multi_concat", True,
        "True for <multiply>. False for <concatenate>")

args = parser.parse_args()
dataset = args.train or args.eval
datapath = os.path.join(FLAGS.vocabs, dataset)


def main():
        data, vocab, htree = get_data(datapath, FLAGS.test_size, FLAGS.val_size)

        train_x, train_y = data[0], data[1]
        val_x, val_y = data[2], data[3]
        test_x, test_y = data[4], data[5]

        idx2word, word2idx = vocab[0], vocab[1]
        # train_x, train_y, val_x, val_y, test_x, test_y, idx2word, word2idx = \
        #       get_data(datapath, FLAGS.test_size, FLAGS.val_size)

        FLAGS.vocab_len = len(idx2word)

        # any key would give the correct max path len
        FLAGS.max_depth = len(htree['<pad>'])
        FLAGS.num_samples = train_x.shape[0]
        FLAGS.sentence_len = train_x.shape[-1]
        FLAGS.timesteps = train_y.shape[-1]
        FLAGS.embedding_method = args.load_embed
        FLAGS.embedding_size = int(args.dim_embed)
        FLAGS.dataset_name = dataset

        # if dataset != "bbc_news":
        #       FLAGS.batch_size = 1

        if args.attlayer == "concat":
                FLAGS.multi_concat = False

        print('news headlines format:', train_y.shape)
        print('news descriptions format:', train_x.shape)
        print('number of tokens in the vocabulary', FLAGS.vocab_len)
        print("huffman tree max depth ", max([len(path) for path in htree.values()]))

        Model = SummarizationModel(FLAGS, idx2word, htree)

        if args.train is not None:
                # Trains the model
                Model.train(train_x, train_y, val_x, val_y)

        else:
                # Evaluates the model
                Model.eval(test_x, test_y)

if __name__ == "__main__":
    main()
