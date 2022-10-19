__author__ = '{Nianheng Wu}'

from sklearn.utils import shuffle
from features import Features
from utils.preprocessor import PrepareData
from utils.dataset import DataProcessor
from statistics import mean
from perceptron import MultiLayerPerceptron
import numpy as np
import argparse


def make_batch(data, batch_size):
    """
    A generator yielding a batch of data of certain size everytime.
    """
    batch_data = []
    for entry in data:
        if entry is not None:
            batch_data.append(entry)
            if len(batch_data) == batch_size:
                yield np.asarray(batch_data)
                batch_data = []
        else:
            continue
    if batch_data:
        yield np.asarray(batch_data)


def train(train_X, train_y, dev_X, dev_y, test_X, test_y):
    """
    The training loop
    """
    # initialze the perceptron with a linear layer, a non-linear layer, and another lear layer
    model = MultiLayerPerceptron()
    model.add_layer((100, parameter_size))
    model.add_activation_func()
    model.add_layer((parameter_size, 2))

    early_stopping = 3
    dev_acces = []

    # the training loop
    for e in range(1, epochs):

        # training each batch
        for train_x_batch, train_y_batch in zip(make_batch(train_X, batch_size), make_batch(train_y, batch_size)):
            model.train(train_x_batch, train_y_batch)

        epoch_dev_acc = []
        for dev_x_batch, dev_y_batch in zip(make_batch(dev_X, batch_size), make_batch(dev_y, batch_size)):
            this_acc = model.validate(dev_x_batch, dev_y_batch)
            epoch_dev_acc.append(this_acc)
        this_dev_acc = mean(epoch_dev_acc)

        dev_acces.append(this_dev_acc)
        print("dev acc: ", this_dev_acc)

        # if the accuracy on dev set does not improve for 3 epochs, stop training
        if len(dev_acces) == early_stopping:
            if dev_acces[2] <= dev_acces[1] and dev_acces[2] <= dev_acces[0]:
                break
            else:
                dev_acces = []
                continue

    predict = model.predict(test_X)
    acc = np.mean(predict == test_y)

    print("acc: ", acc, "acc on dev:", this_dev_acc, "epochs: ", e)


def data_prepare():
    """
    Prepare the data for training, validation and testing.
    """

    # This is a binary classification problem, and we are creating binary examples here.
    data_processor = DataProcessor()
    train_examples, train_labels = data_processor.create_binary_examples(file_path=train_json_path,
                                                                         input_type='full-seq',
                                                                         labels_path=train_json_labels)
    dev_examples, dev_labels = data_processor.create_binary_examples(file_path=dev_json_path, input_type='full-seq',
                                                                     labels_path=dev_json_labels)
    test_examples, test_labels = data_processor.create_binary_examples(file_path=test_json_path, input_type='full-seq',
                                                                       labels_path=test_json_labels)
    # text preprocessing: exp. remove non-word tokens, etc.
    processor = PrepareData()
    train_normalized = processor.preprocess_data(train_examples, flg_clean=True, flg_stemm=False, flg_lemm=False)
    dev_normalized = processor.preprocess_data(dev_examples, flg_clean=True, flg_stemm=False, flg_lemm=False)
    test_normalized = processor.preprocess_data(test_examples, flg_clean=True, flg_stemm=False, flg_lemm=False)

    # get word embeddings for all words
    train_feature_extractor = Features(embedding_file=embed+'/glove.6B.100d.txt', lst_corpus=train_normalized,
                                       embedding_dim=100)
    dev_feature_extractor = Features(embedding_file=embed+'/glove.6B.100d.txt', lst_corpus=dev_normalized, embedding_dim=100)
    test_feature_extractor = Features(embedding_file=embed+'/glove.6B.100d.txt', lst_corpus=test_normalized, embedding_dim=100)
    train_sentence_vectors = train_feature_extractor.create_features()
    dev_sentence_vectors = dev_feature_extractor.create_features()
    test_sentence_vectors = test_feature_extractor.create_features()

    # randomize the samples
    train_sentence_vectors, train_labels = shuffle(train_sentence_vectors, train_labels)

    # samples are ready
    train_X = train_sentence_vectors
    train_y = train_labels
    dev_X = dev_sentence_vectors
    dev_y = dev_labels
    test_X = test_sentence_vectors
    test_y = test_labels

    return train_X, train_y, dev_X, dev_y, test_X, test_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='path to the training set')
    parser.add_argument('--train_lbs', help='path to the training labels')
    parser.add_argument('--dev', help='path to the development set')
    parser.add_argument('--dev_lbs', help='path to the development labels')
    parser.add_argument('--test', help='path to the test set')
    parser.add_argument('--test_lbs', help='path to the test labels')
    parser.add_argument('--embed', help='path to the embedding files')
    parser.add_argument('--parameter_size', help='parameter size')
    parser.add_argument('--batch_size', help='batch size')
    parser.add_argument('--epochs', help='maximum epochs')
    args = parser.parse_args()

    # define path to training samples
    train_json_path = args.train
    train_json_labels = args.train_lbs
    dev_json_path = args.dev
    dev_json_labels = args.dev_lbs
    test_json_path = args.test
    test_json_labels = args.test_lbs
    embed = args.embed
    parameter_size = args.parameter_size
    batch_size = args.batch_size
    epochs = args.epochs

    # prepare the data and start training
    train(data_prepare())
