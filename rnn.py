from __future__ import absolute_import, division, print_function

import os
from builtins import (str, open, range)
from time import time

import numpy as np
from keras.layers import Dense, TimeDistributed, GRU, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

import parameters as p


def preprocess_data(sen_space):
    # il ajoute un & au debut et \n a la fin de chaque smile et separe chaque lettre dans une liste
    # les atomes entre [ ] restent dans la meme chaine

    # print sen_space
    all_smile = []
    end = "\n"
    element_table = ["C", "N", "B", "O", "P", "S", "F", "Cl", "Br", "I", "(", ")", "=", "#"]

    for i in range(len(sen_space)):
        # word1=sen_space[i]
        word_space = sen_space[i]
        word = []
        # word_space.insert(0,end)
        j = 0
        while j < len(word_space):
            word_space1 = []
            # word_space1.append(word_space[j])
            if word_space[j] == "[":
                word_space1.append(word_space[j])
                j = j + 1
                while word_space[j] != "]":
                    word_space1.append(word_space[j])
                    j = j + 1
                word_space1.append(word_space[j])
                word_space2 = ''.join(word_space1)
                word.append(word_space2)
                j = j + 1
            else:
                word_space1.append(word_space[j])

                if j + 1 < len(word_space):
                    word_space1.append(word_space[j + 1])
                    word_space2 = ''.join(word_space1)
                else:
                    word_space1.insert(0, word_space[j - 1])
                    word_space2 = ''.join(word_space1)

                if word_space2 not in element_table:
                    word.append(word_space[j])
                    j = j + 1
                else:
                    word.append(word_space2)
                    j = j + 2

        word.append(end)
        word.insert(0, "&")
        all_smile.append(list(word))

    val = ["\n"]
    for smile in all_smile:
        for a in smile:
            if a not in val:
                val.append(a)

    return val, all_smile


def convert_data_to_numbers(vocabulary, smiles):
    """
    Prepare data for the RNN,
    convert smiles into arrays of number corresponding to their index in the vocabulary array
    """
    x_train = [[vocabulary.index(atom) for atom in smile] for smile in smiles]
    y_train = [x[1:] + [0] for x in x_train]
    # y_train = x_train.copy()
    return x_train, y_train


def load_model():
    # load tree, info, data and model
    if os.path.isfile(p.f_rnn_architecture) and os.path.isfile(p.f_rnn_weights):
        with open(p.f_rnn_architecture, 'r') as f:
            p.model = model_from_json(f.read())
            # Load weights into the new model
            p.model.load_weights(p.f_rnn_weights)
            print("Loaded model from disk")
    else:
        if not p.train_rnn:
            raise Exception("No model founded, you have to train it first")


def create_rnn_model(smiles):
    p.vocabulary, all_smile = preprocess_data(smiles)

    # free some memory...
    del smiles

    vocab_size = len(p.vocabulary)

    print("Vocabulary length = %d" % vocab_size)
    print("vocabulary = %s " % str(p.vocabulary))

    x_train, y_train = convert_data_to_numbers(p.vocabulary, all_smile)

    print("Smiles converted to numbers")

    maxlen = 81

    x = sequence.pad_sequences(x_train, maxlen=maxlen, dtype='int32',
                               padding='post', truncating='pre', value=0.)
    y = sequence.pad_sequences(y_train, maxlen=maxlen, dtype='int32',
                               padding='post', truncating='pre', value=0.)

    print("Loading y_train_one_hot")

    y_train_one_hot = np.array([to_categorical(sent_label, num_classes=vocab_size) for sent_label in y])
    print(y_train_one_hot.shape)

    n = x.shape[1]

    p.model = Sequential()

    p.model.add(Embedding(input_dim=vocab_size, output_dim=vocab_size, input_length=n, mask_zero=False))
    p.model.add(GRU(units=256, return_sequences=True, activation="tanh", input_shape=(81, 26)))
    p.model.add(Dropout(0.2))
    p.model.add(GRU(256, activation='tanh', return_sequences=True))
    p.model.add(Dropout(0.2))
    p.model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    optimizer = Adam(lr=0.01)
    print(p.model.summary())

    log = p.f_rnn_logs
    if not os.path.isdir(log):
        os.mkdir(log)
    tensorboard = TensorBoard(log_dir=log + "/{}".format(time()))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(p.f_rnn_weights, save_best_only=True, monitor='val_loss', mode='min')

    p.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    p.model.fit(x, y_train_one_hot, epochs=2, batch_size=512, validation_split=0.1,
                callbacks=[tensorboard, early_stopping, mcp_save])

    model_json = p.model.to_json()
    with open(p.f_rnn_architecture, "w") as json_file:
        json_file.write(model_json)
    print("Model saved")

    p.model.load_weights(p.f_rnn_weights)


# if __name__ == "__main__":
#     from sklearn.utils import resample
#     from main import load_smiles
#     load_smiles()
#     boots = []
#     for i in range(10):
#         print(i)
#         boots.append(resample(p.smiles, replace=False, n_samples=100000, random_state=i))


