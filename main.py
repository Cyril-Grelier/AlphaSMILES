from __future__ import print_function

import json
import os
import pickle

import mcts
import parameters as p
import rnn
import scorer
from csv_to_dict import load_dict
from node import Node


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


def load_data():
    if not os.path.isfile(p.f_data_json):
        print("Generating data...")
        data = load_dict(atom_allowed=p.atom_allowed, char_not_allowed=p.char_not_allowed)
        with open(p.f_data_json, 'w') as f:
            json.dump(data, f)
    else:
        print("Loading data...")
        with open(p.f_data_json, 'r') as f:
            data = json.load(f)
    p.smiles = list(data.keys())
    with open(p.f_smiles, "w") as s_txt:
        for s in p.smiles:
            s_txt.write(s)
            s_txt.write("\n")
    # Save the scores
    logP = []
    sa = []
    cy = []
    for d in data:
        logP.append(data[d]["logp"])
        sa.append(data[d]["sa"])
        cy.append(data[d]["cycle"])
    with open(p.f_logp, "w") as logp_f:
        for lo in logP:
            logp_f.write(str(lo) + "\n")

    with open(p.f_sa_scores, "w") as s:
        for lo in sa:
            s.write(str(lo) + "\n")

    with open(p.f_cycles_scores, "w") as c:
        for lo in cy:
            c.write(str(lo) + "\n")


def load_smiles():
    if os.path.isfile(p.f_smiles):
        p.smiles = []
        with open(p.f_smiles) as f_s:
            for s in f_s:
                p.smiles.append(s)
        print("SMILES loaded from disk")
        print("Using " + str(len(p.smiles)) + " SMILES")
    else:
        raise Exception("Error you have to load data at least once")


def train_rnn():
    # Create the RNN Model
    p.model, p.vocabulary = rnn.create_rnn_model(smiles=p.smiles)


def load_rnn():
    rnn.load_model()


def load_parameters_mcts():
    if os.path.isfile(p.f_mcts_tree_pckl) \
            and os.path.isfile(p.f_data_dict_json) \
            and os.path.isfile(p.f_info_json):
        print("Loading previous tree, data and info")
        with open(p.f_mcts_tree_pckl, 'rb') as f:
            pickler = pickle.Unpickler(f)
            p.tree = pickler.load()
        with open(p.f_data_dict_json, 'r') as f:
            p.data = json.load(f)
        with open(p.f_info_json, 'r') as f:
            p.tree_info = json.load(f)
    else:
        print("New tree")
        p.tree_info = dict()
        p.tree_info[p.info_created] = 0
        p.tree_info[p.info_good] = 0
        p.tree_info[p.info_alrd_tested] = 0
        p.data = dict()
        p.tree = Node()
    if p.use_dft:
        p.scorer = scorer.ScorerDFT(alpha=1)
    else:
        p.scorer = scorer.ScorerValidSMILES(alpha=1)
    mcts.reset_score_visit(p.tree)
    mcts.load_scores()
    p.tree.print()


if __name__ == "__main__":
    # prefix btx
    # prefix = 'c1c2c(=O)n(C)c(=O)c(cc3)c2c4c3c2ccccc2sc4c1'
    # prefix = ['c', '1', 'c', '2', 'c', '(', '=', 'O', ')', 'n', '(', 'C', ')', 'c', '(', '=', 'O', ')', 'c', '(', 'c',
    #           'c', '3', ')', 'c', '2', 'c', '4', 'c', '3', 'c', '2', 'c', 'c', 'c', 'c', 'c', '2', 's', 'c', '4', 'c',
    #           '1']

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())

    if p.load_data:
        load_data()
    else:
        if p.train_rnn:
            load_smiles()

    if p.train_rnn:
        train_rnn()
    else:
        load_rnn()

    if p.run_mcts:
        load_parameters_mcts()

        print("Lenght data already tested : %d" % len(p.data))

        print("Working with :")
        print("Vocabulary : %s" % str(p.vocabulary))
        print("Len(vocabulary) : %d" % len(p.vocabulary))
        mcts.launch(nb_turn=500)
