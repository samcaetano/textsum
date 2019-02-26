#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Using the BBC News dataset (@ http://mlg.ucd.ie/datasets/bbc.html) as a toy dataset
from __future__ import print_function
from collections import Counter
from itertools import chain
from os import listdir
from nltk.tokenize import RegexpTokenizer
import cPickle as pickle
import pandas as pd
import numpy as np
import string
import glob
import os
import queue
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Train the model")

arg = parser.parse_args()
T = RegexpTokenizer(r"[a-zA-Z]+")

class HuffmanNode(object):
    def __init__(self, left=None, right=None, root=None):
        self.left = left
        self.right = right
        self.root = root     # Why?  Not needed for anything.
    def children(self):
        return((self.left, self.right))

def create_tree(frequencies):
    p = queue.PriorityQueue()
    for value in frequencies:    # 1. Create a leaf node for each symbol
        p.put(value)             #    and add it to the priority queue
    while p.qsize() > 1:         # 2. While there is more than one node
        l, r = p.get(), p.get()  # 2a. remove two highest nodes
        node = HuffmanNode(l, r) # 2b. create internal node with children
        p.put((l[0]+r[0], node)) # 2c. add new node to queue
    return p.get()

def walk_tree(node, prefix="", code={}):
        if isinstance(node[1].left[1], HuffmanNode):
                walk_tree(node[1].left, prefix + "0", code)
        else:
                code[node[1].left[1]] = prefix + "0"

        if isinstance(node[1].right[1], HuffmanNode):
                walk_tree(node[1].right, prefix + "1" , code)
        else:
                code[node[1].right[1]] = prefix+"1"
        return(code)

def unify_datasets():
        # Load all toy dataset and put all together
        folders = ["{}".format(_) for _ in range(1,6)]
        filenames = []
        for folder in folders:
                for filename in glob.glob(folder+"/*.txt"):
                        filenames.append(filename)
        return filenames

def load_data(files):
        headings, descr = [], []
        for file in files:
                with open(file) as f:
                        headings.append([f.readline()])
                        descr.append(f.readlines())

        descr = [[" ".join(content)] for content in descr]
        for i, (x, y) in enumerate(zip(descr, headings)):
                headings[i] = [T.tokenize(y[0].lower())]
                descr[i] = [T.tokenize(x[0].lower())]

        return headings, descr

def tokenizer(sentence):
        sentence = re.sub(r"[^a-zA-Z]+", " ", sentence)
        return sentence.split()

def load_cnn_dm(directory):
        def load_doc(filename):
                # open the file as read only
                file = open(filename)
                # read all text
                text = file.read()
                # close the file
                file.close()
                return text

        # split a document into news story and highlights
        def split_story(doc):
                # find first highlight
                index = doc.find("@highlight")
                # split into story and highlights
                story, highlights = doc[:index], doc[index:].split("@highlight")
                # strip extra white space around each highlight
                highlights = [h.strip() for h in highlights if len(h) > 0]
                return story, highlights

        # load all stories in a directory
        def load_stories(directory):
                stories = list()
                for name in listdir(directory)[:5000]:
                        filename = directory + "/" + name
                        # load document
                        doc = load_doc(filename)
                        # split into story and highlights
                        story, highlights = split_story(doc)
                        # store
                        stories.append({"story":story, "highlights":highlights})

                return stories

        stories = load_stories(directory)

        headings, descr = [], []
        for example in stories:
                s = " ".join(example["story"].split("\n")).lower()
                #s = T.tokenize(s)
                s = tokenizer(s)

                try:
                        s = s[s.index("cnn")+1:]
                except ValueError:
                        pass

                h = " ".join(example["highlights"]).lower()
                #h = T.tokenize(h)
                h = tokenizer(h)

                headings.append([h])
                descr.append([s])

        return headings, descr

def get_vocab(lst):
        vocabcount = Counter(token for doc in lst for sentence in doc
                for token in sentence)
        vocabcount.update({'<pad>': 1, '<go>': 1, '<eos>': 1})

        print("vocab original size ", len(vocabcount))
        vocab = map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1]))

        assert len(vocab) == len(vocabcount), "Something wrong in vocab"

        return vocab, vocabcount

def get_idx(vocab):
        #empty = 0
        #go = 1
        #eos = 2
        #start_idx = 3


        word2idx = dict((word, idx) for idx,word in enumerate(vocab))
        #word2idx['<pad>'] = empty
        #word2idx['<go>'] = go
        #word2idx['<eos>'] = eos

        idx2word = dict((idx, word) for word, idx in word2idx.items())

        return word2idx, idx2word

def main():
        if arg.dataset == "bbc_news":
                files = unify_datasets()
                headings, descr = load_data(files)

        elif arg.dataset == "cnn_dm":
                cnn_h, cnn_d = load_cnn_dm("cnn/stories/")
                print("Loaded CNN stories")

                dm_h, dm_d = load_cnn_dm("dailymail/stories/")
                print("Loaded DailyMail stories")

                headings = cnn_h + dm_h
                descr = cnn_d + dm_d

        assert len(headings) == len(descr), "No pair set"

        print(headings[0], descr[0])

        vocab, vocabcount = get_vocab(headings+descr)

        print("Building vocab dictionary")
        word2idx, idx2word = get_idx(vocab)

        freqs = [(float(v)/len(vocabcount), k) for k,v in vocabcount.iteritems()]

        print("Building huffman tree")
        node = create_tree(freqs)
        code = walk_tree(node)

        X, Y = [], []

        for head, content in zip(headings, descr):
                X.append([word2idx[token] for token in content[0]])
                Y.append([word2idx[token] for token in head[0]])

        for heading in Y:
                heading.insert(0, word2idx['<go>'])
                heading.append(word2idx['<eos>'])

        #print([idx2word[t] for t in X[0]], [idx2word[t] for t in Y[0]])

        assert len(descr) == len(X), "Wrong dimensions on X"

        with open('../vocabs/'+arg.dataset+'.vocab.pkl','wb') as fp:
                pickle.dump((idx2word, word2idx),fp,-1)

        with open('../vocabs/'+arg.dataset+'.vocab.data.pkl','wb') as fp:
            pickle.dump((X,Y),fp,-1)
            
        with open('../vocabs/'+arg.dataset+'.tree.pkl', 'wb') as fp:
                pickle.dump(code, fp, -1)

main()
