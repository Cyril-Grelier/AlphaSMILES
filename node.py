from __future__ import print_function

import pptree

import parameters as p
from smiles import SMILES


class Node:

    def __init__(self, smiles=SMILES(), parent=None):
        self.smiles = smiles
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0.0

    def new_child(self, child_smile):
        self.children.append(Node(child_smile, self))

    def update(self, reward):
        self.score += reward
        self.visits += 1
        if self.parent:
            self.parent.update(reward)

    def fully_expanded(self):
        return (len(self.children) == (len(p.vocabulary) - 2)) or (len(self.smiles.element) > 80)

    def out_pptree(self, parent=None):
        name = repr(self)
        # str(self.score) + " " + str(self.smiles) + " " + str(self.visits)
        if parent:
            current = pptree.Node(name, parent)
        else:
            current = pptree.Node(name)
        for c in self.children:
            c.out_pptree(current)
        return current

    def print(self):
        pptree.print_tree(self.out_pptree())

    def __repr__(self):
        return str(self.smiles) + " " + str(int(self.visits)) + " " + str(round(self.score, 2))
