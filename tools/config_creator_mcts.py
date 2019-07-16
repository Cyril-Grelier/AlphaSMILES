import json

'''
This code help you to prepare the json file of configuration to train the rnn
'''

config = dict()

# name of your configuration
config['configuration_name'] = "generated"

# rnn directory (the name of the rnn configuration)
config['rnn_repertory'] = "rnn_test"

# long prefix, the RNN and MCTS won't see it until the properties are calculated
# use it if the prefix is really long (more than 15 tokens approximately)
config['long_prefix'] = ['c', '1', 'c', '2', 'c', '(', '=', 'O', ')', 'n', '(', 'C', ')', 'c', '(', '=', 'O', ')',
                         'c', '(', 'c', 'c', '3', ')', 'c', '2', 'c', '4', 'c', '3', 'c', '2', 'c', 'c', 'c', 'c',
                         'c', '2', 's', 'c', '4', 'c', '1']

# short prefix, use a string to represent the prefix and use it as the first node of the tree
# represent the prefix like this : ['c', '1']
# note : all tokens present in the prefix have to be in the RNN tokens.
config['prefix'] = []
config['from'] = 'node'  # 'node' or 'root'

config['SMILES_simulated_per_node'] = 2

config['nb_turn'] = 200

config['exploration_vs_exploitation'] = 1

config['expansion'] = "proba"
config['proba_min'] = 0.0001

# multitasking
config['n_jobs'] = 4
config['nb_core_dft'] = 4

config['properties'] = [("mcts.properties.properties", "SAScoreProperty2DDecorator"),
                        ("mcts.properties.properties", "CycleProperty2DDecorator"),
                        ("mcts.properties.properties", "LogPProperty2DDecorator"),
                        ("mcts.properties.properties", "DFTPropertyDecorator"),
                        ]

config['scorer'] = ("mcts.scorer.scorer", 'ScorerValidSMILES')
config['alpha_scorer'] = 1

with open('../mcts/configurations/' + config['configuration_name'] + '.json', 'w') as conf:
    json.dump(config, conf)
