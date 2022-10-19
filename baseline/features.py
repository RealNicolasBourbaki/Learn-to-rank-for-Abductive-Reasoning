__author__ = '{Esra Dönmez}'

import os
import numpy as np


class Features:
    """
    A class for future extraction.
    ...
    Param:
        embedding_file: GloVe embedding file
        lst_corpus: list of training examples
        embedding_dim: embedding dimensions
    """
  
    def __init__(self, embedding_file, lst_corpus, embedding_dim):
        self.embedding_file = embedding_file
        self.lst_corpus = lst_corpus
        self.embedding_dim = embedding_dim

    def get_word_index(self):
        """
        Creates a word index from the corpus.
        """
        vocab = set()
        for l in self.lst_corpus:
            for t in l:
                vocab.add(t)

        word_index = dict()
        for i, w in enumerate(vocab):
            word_index[w] = i

        return word_index

    def create_embedding_index(self):
        """
        Creates an embedding index for GloVe embeddings.
        """
        embeddings_index = {}
        f = open(os.path.join('./', self.embedding_file))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        # print('Found %s word vectors.' % len(embeddings_index))
        return embeddings_index

    def create_embedding_matrix(self):
        """
        Creates an embedding matrix given the word index and the embedding index.
        """
        word_index = self.get_word_index()
        embeddings_index = self.create_embedding_index()
        embedding_matrix = np.zeros((len(word_index) + 1, self.embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def create_features(self):
        """
        Creates the GloVe words embeddings from the training data.
        Averages the word embeddings to create an averaged bag of words.
        """
        embedding_matrix = self.create_embedding_matrix()
        word_index = self.get_word_index()
        sentence_embeddings = []
        for s in self.lst_corpus:
            summed = np.zeros(self.embedding_dim)
            for w in s:
                w_embedding = embedding_matrix[word_index.get(w)]
                summed += w_embedding
            sentence_embeddings.append(summed / len(s))

        return sentence_embeddings
