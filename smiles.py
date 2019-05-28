from __future__ import absolute_import, division, print_function

import hashlib
from builtins import (str)

import networkx as nx
import numpy as np
from keras.preprocessing import sequence
from rdkit import RDLogger
from rdkit.Chem import Descriptors, rdmolops, MolFromSmiles

import parameters as p
import sascorer
from dft import calcul_dft

# to change RDKit logs
# RDLogger.logger().setLevel(RDLogger.DEBUG)
RDLogger.logger().setLevel(RDLogger.CRITICAL)


class SMILES:

    def __init__(self, element=None):
        self.element = element if element else []
        if element and element[-1] == '\n':
            self.properties = dict()
            self.scores = dict()
        else:
            self.properties = dict()
            self.scores = None

    def next_atoms(self):
        smiles_int = mol_to_int(['&'] + self.element)
        x = np.reshape(smiles_int, (1, len(smiles_int)))
        x_pad = sequence.pad_sequences(x, maxlen=81, dtype='int32',
                                       padding='post', truncating='pre', value=0.)
        predictions = p.model.predict(x_pad)
        preds = np.asarray(predictions[0][len(smiles_int) - 1]).astype('float64')
        # for i in range(len(p.vocabulary)):
        #     print(str(p.vocabulary[i]) + "\t\t%20f" % preds[i])
        # print("preds")
        # print(preds)
        smiles_to_expand = []
        for i, pred in enumerate(preds):
            '''
            for the first node:     with 0.0001 10/21 no accepted
            \n      0.000069        no
            &       0.000000        no
            C       0.844086
            O       0.080062
            (       0.000476
            =       0.000779
            )       0.000053        no
            c       0.010339
            1       0.000086        no
            N       0.058904
            n       0.000297
            2       0.000054        no
            3       0.000015        no
            4       0.000002        no
            [nH]    0.000020        no
            Cl      0.001153
            S       0.000430
            o       0.000019        no
            #       0.000123
            [NH]    0.003027
            s       0.000006        no
            '''
            if pred > 0.0001 and p.vocabulary[i] != "&":
                smiles_to_expand.append(SMILES(int_to_smile(smiles_int[1:]) + [p.vocabulary[i]]))
        return smiles_to_expand

    def next_atom(self):
        smiles_int = mol_to_int(['&'] + self.element)
        x = np.reshape(smiles_int, (1, len(smiles_int)))
        x_pad = sequence.pad_sequences(x, maxlen=81, dtype='int32',
                                       padding='post', truncating='pre', value=0.)
        predictions = p.model.predict(x_pad)
        preds = np.asarray(predictions[0][len(smiles_int) - 1]).astype('float64')

        # for i in range(len(p.vocabulary)):
        #     print(str(p.vocabulary[i]) + "\t\t%20f" % preds[i])
        # print("preds")
        # print(preds)
        preds = preds[2:]
        preds = preds / np.sum(preds)
        next_probas = np.random.multinomial(1, preds, 1)
        # print("next_probas")
        # print(next_probas)
        next_int = np.argmax(next_probas) + 2
        smiles_int.append(next_int)
        return SMILES(int_to_smile(smiles_int[1:]))
        # return SMILES(self.element + [random.choice(p.vocabulary[2:])])

    def end_smiles(self):
        smiles_int = mol_to_int(['&'] + self.element)
        while not smiles_int[-1] == p.vocabulary.index("\n"):
            x = np.reshape(smiles_int, (1, len(smiles_int)))
            x_pad = sequence.pad_sequences(x, maxlen=81, dtype='int32',
                                           padding='post', truncating='pre', value=0.)
            predictions = p.model.predict(x_pad)
            preds = np.asarray(predictions[0][len(smiles_int) - 1]).astype('float64')
            preds = preds / np.sum(preds)
            next_probas = np.random.multinomial(1, preds, 1)
            next_int = np.argmax(next_probas)
            smiles_int.append(next_int)
            if len(smiles_int) > 81:
                break
        return SMILES(int_to_smile(smiles_int[1:]))

    def terminal(self):
        return (self.element[-1] == '\n') if self.element else False

    def calcul_properties(self):
        """
        Calcul logp, sa score, cycle score, dft
        """
        molecule = MolFromSmiles("".join(self.element[:-1]))
        if not molecule:
            self.properties[p.s_valid] = False
            return
        self.properties[p.s_valid] = True
        try:
            self.properties[p.s_logp] = Descriptors.MolLogP(molecule)
        except Exception as e:
            print("!" * 100)
            print(e)
        self.properties[p.s_sa] = sascorer.calculate_score(molecule)
        cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(molecule)))
        self.properties[p.s_cycle] = max([len(j) for j in cycle_list]) if cycle_list else 0
        self.properties[p.s_id] = p.tree_info[p.info_good]
        if p.use_dft:
            self.properties[p.s_dft] = calcul_dft(self.properties[p.s_id], "".join(self.element[:-1]))

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(hashlib.sha256("".join(self.element).encode('utf-8')).hexdigest())

    def __repr__(self):
        if self.element and self.element[-1] == '\n':
            return "'" + "".join(self.element[:-1]) + "'"
        else:
            return "_" + "".join(self.element) + "_"


def int_to_smile(list_of_int):
    return [p.vocabulary[s] for s in list_of_int]


def mol_to_int(list_of_mol):
    return [p.vocabulary.index(s) for s in list_of_mol]
