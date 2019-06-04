from __future__ import absolute_import, division, print_function

import json
import math
import pickle
import random
import time
from builtins import (str, open, range, int)

from joblib import Parallel, delayed

import parameters as p
from smiles import SMILES


def stop_next_turn():
    with open(p.f_stop_mcts) as f:
        if f.read() == "stop" or f.read() == "stop\n":
            print("MCTS stopped with signal 'stop' in %s file" % p.f_stop_mcts)
            return True
        return False


def launch(nb_turn=1):
    print("Let's begin")
    start = time.time()
    node = p.tree
    i = 0
    while (i < nb_turn) and (not stop_next_turn()):
        print("Turn %d" % i)
        node_to_expand = selection(node)
        print("Node to expand : " + str(node_to_expand))
        new_node = expansion(node_to_expand)
        print("New nodes : " + str(new_node))
        new_smiles = simulation(new_node)
        print("new smiles : " + str(new_smiles))
        update(new_smiles)  # (new_node, new_smiles)
        save_tree(node)
        save_data_and_info()
        i += 1
    print("MCTS worked during %d s" % int(time.time() - start))
    print("Found %d valid molecules (%d already tested) out of %d generated" % (
        p.tree_info[p.info_good], p.tree_info[p.info_alrd_tested],
        p.tree_info[p.info_created]))


def selection(node):
    while not node.smiles.terminal():
        if not node.children:
            return node
        else:
            node = ubc(node)
    return node


def ubc(node):
    if not node.children:
        raise Exception("ubc impossible, no child : " + repr(node))
    best_score = -float("inf")
    best_children = []
    for c in node.children:
        if not c.smiles.terminal():
            exploit = c.score / c.visits
            explore = math.sqrt(2.0 * math.log(node.visits) / float(c.visits))
            score = exploit + p.exporation_vs_explotation * explore
            if score == best_score:
                best_children.append(c)
            if score > best_score:
                best_children = [c]
                best_score = score
    if not best_children:
        return node
    return random.choice(best_children)


def expansion(nodes_to_expand):
    new_nodes = []
    for s in nodes_to_expand.smiles.next_atoms():
        nodes_to_expand.new_child(s)
        new_nodes.append(nodes_to_expand.children[-1])
    return new_nodes


def simulation(nodes):
    all_new_smiles = []
    for n in nodes:
        new_smiles = []
        while len(new_smiles) < 2:
            smiles = n.smiles.end_smiles()
            new_smiles.append(smiles)
        all_new_smiles += new_smiles
    return all_new_smiles


def update_smiles(smiles):
    already = False
    if repr(smiles) in p.data.keys():
        already = True
        with p.lock_update_data:
            p.tree_info[p.info_alrd_tested] += 1
        smiles.properties = p.data[repr(smiles)]
    else:
        smiles.calculation_of_properties()
        with p.lock_update_data:
            p.data[repr(smiles)] = smiles.properties
    reward = p.scorer.reward(p.data[repr(smiles)], already)
    with p.lock_update_node:
        node = get_node_starting_with("".join(smiles.element))
        print("Reward %s : %f on node %s" % (str(smiles), reward, repr(node)))
        node.update(reward)


def update(new_smiles):
    p.tree_info[p.info_created] += len(new_smiles)
    print("Update")
    print("%d smiles to process" % len(new_smiles))
    backend = 'threading'  # 'loky' 'threading' 'multiprocessing'
    Parallel(n_jobs=p.n_jobs, backend=backend)(delayed(update_smiles)(s) for s in new_smiles)


# def update(new_nodes, new_smiles):
#     p.tree_info[p.info_created] += len(new_smiles)
#     print("Update")
#     print("%d smiles to process" % len(new_smiles))
#
#     backend = 'threading'  # 'loky' 'threading' 'multiprocessing'
#
#     Parallel(n_jobs=p.n_jobs, backend=backend)(
#         delayed(update_node_with_smile)(n, new_smiles[j + 2 * i]) for i, n in enumerate(new_nodes) for j in range(2))
#
#     # for i, n in enumerate(new_nodes):
#     #     for j in range(2):
#     #         print("SMILES %d/%d" % (j + 2 * i + 1, len(new_smiles)))
#     #         update_node_with_smile(n, new_smiles[j + 2 * i])
#
#
# def update_node_with_smile(node, smiles):
#     already = False
#     if repr(smiles) in p.data.keys():
#         already = True
#         p.tree_info[p.info_alrd_tested] += 1
#         smiles.properties = p.data[repr(smiles)]
#     else:
#         smiles.calculation_of_properties()
#         with p.lock_update_data:
#             p.data[repr(smiles)] = smiles.properties
#     if smiles.properties[p.s_valid]:
#         p.tree_info[p.info_good] += 1
#     with p.lock_update_node:
#         reward = p.scorer.reward(p.data[repr(smiles)], already)
#         print("Reward %s : %f on node %s" % (str(smiles), reward, str(node)))
#         node.update(reward)
#         print("Reward done : %s" % str(node))


def get_node_with_prefix(node, smiles):
    """
    be carefull, create nodes until the smiles is complete in the tree
    :param node:
    :param smiles:
    :return:
    """
    current_node = node
    for i in range(len(smiles.element)):
        next_node = None
        for n in current_node.children:
            if n.smiles.element == smiles.element[:i + 1]:
                next_node = n
                break
        if next_node:
            current_node = next_node
        else:
            current_node.new_child(SMILES(smiles.element[:i + 1]))
            current_node = current_node.children[-1]
    return current_node


def save_tree(node):
    with open(p.f_mcts_tree_pckl, "wb") as pf:
        pickler = pickle.Pickler(pf)
        pickler.dump(node)
        print("Tree saved")


def load_tree():
    with open(p.f_mcts_tree_pckl, 'rb') as pf:
        pickler = pickle.Unpickler(pf)
        node = pickler.load()
    return node


def save_data_and_info():
    with open(p.f_data_dict_json, 'w') as f:
        json.dump(p.data, f)
    with open(p.f_info_json, 'w') as f:
        json.dump(p.tree_info, f)


def get_node_starting_with(smiles):
    current_node = p.tree
    s = smiles
    if (smiles[-1] == "\n") or (smiles[-1] == "'"):
        s = s[:-1]
    if smiles[0] == "'":
        s = s[1:]
    i = 0
    while current_node.children and not (repr(current_node.smiles)[1:-1] == s):
        children = current_node.children
        children.sort(key=lambda x: len(str(x.smiles)), reverse=True)
        for c in children:
            c_smiles = str(c.smiles)[1:-1]
            len_c_smiles = len(c_smiles)
            if s[:len_c_smiles] == c_smiles:
                current_node = c
                break
    return current_node


def load_scores():
    print("Loading scores")
    for s in p.data.keys():
        node = get_node_starting_with(s[1:])
        node.update(p.scorer.reward(p.data[s], False))


def reset_score_visit(node):
    node.visits = 0
    node.score = 0.0
    if node.children:
        for c in node.children:
            reset_score_visit(c)

# if __name__ == "__main__":
#     # to test get_node_with_prefix
#     node = Node()
#     node.new_child(SMILES(['O']))
#     node.children[0].new_child(SMILES(node.children[0].smiles.element + ['N']))
#     node.children[0].new_child(SMILES(node.children[0].smiles.element + ['C']))
#     node.children[0].children[1].new_child(SMILES(node.children[0].children[1].smiles.element + ['c']))
#     node.children[0].children[1].new_child(SMILES(node.children[0].children[1].smiles.element + ['F']))
#     node.children[0].children[1].new_child(SMILES(node.children[0].children[1].smiles.element + ['D']))
#     node.children[0].new_child(SMILES(node.children[0].smiles.element + ['c']))
#     node.children[0].new_child(SMILES(node.children[0].smiles.element + ['F']))
#     node.new_child(SMILES(node.smiles.element + ['C']))
#     node.new_child(SMILES(node.smiles.element + ['N']))
#     node.new_child(SMILES(node.smiles.element + ['[NH]']))
#     pptree.print_tree(node.out_pptree())
#     prefix = get_node_with_prefix(node, SMILES(['O', 'C', 'F', 'G']))
#     pptree.print_tree(node.out_pptree())
#     pptree.print_tree(prefix.out_pptree())
