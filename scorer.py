from __future__ import absolute_import, division, print_function

import math
from abc import ABC, abstractmethod
from builtins import (super)

import parameters as p


class Scorer(ABC):

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    @abstractmethod
    def score(self, smiles):
        pass

    def reward(self, smiles, already):
        score = self.score(smiles)
        reward = ((self.alpha * score) / (1 + math.fabs(self.alpha * score)))
        # return score
        return reward if not already or not smiles[p.s_valid] else (reward / 2)


class ScorerValidSMILES(Scorer):

    def __init__(self, alpha=0.5):
        super().__init__(alpha)

    def score(self, smiles):
        return 1 if smiles[p.s_valid] else -1


class ScorerDFT(Scorer):

    def __init__(self, alpha=0.5):
        super().__init__(alpha)

    def score(self, smiles):
        if smiles[p.s_valid]:
            try:
                dft = smiles[p.s_dft]
            except Exception as e:
                print(e)
                return -0.5
            score = 0
            for line in dft:
                if 500 < line["nm"] < 1000:
                    score += line["f"] * 100
            return score
        else:
            return -100

    # def calcul_score(self):
    #     # voir https://www.desmos.com/calculator
    #     # -\frac{x+\left(-2.97\right)}{0.73}    <- valeur de base
    #     # -\frac{x+\left(-4\right)}{0.5}        <- valeurs estimees bonnes
    #     # if self.properties[p.s_valid]:
    #     #     p.tree_info[p.info_good] += 1
    #     #     sa_score = -(self.properties[p.s_sa] + p.sa_scores_mean) / p.sa_scores_std
    #     #     self.scores[p.s_sa] = sa_score
    #     return 1 if self.properties[p.s_valid] else -1
    #
    # def reward(self):
    #     return 1 if self.properties[p.s_valid] else -1
