import numpy as np

'''
Parameters
if you change the vocabulary update it here too
'''

load_data = False
train_rnn = False
run_mcts = True
use_dft = False

# files names

#       data
f_data_csv = 'data/data.csv'
f_smiles = 'data/smiles'
f_cycles_scores = 'data/cycles_scores'
f_logp = 'data/logp'
f_sa_scores = 'data/sa_scores'
f_data_json = 'data/CHON_Cl_S.json'

#       rnn
f_rnn_architecture = 'RNN_Model/model_architecture.json'
f_rnn_weights = 'RNN_Model/model_weights.h5'
f_rnn_logs = 'RNN_Model/logs'

#       mcts
f_node_pckl = 'MCTS/nodes.pckl'
f_mcts_pckl = 'MCTS/tree.pckl'
f_mcts_tree_pckl = 'MCTS/mcts_tree.pckl'
f_stop_mcts = 'MCTS/stop_mcts'

#       dft
r_dft = 'generated/dft/'

#       generated molecules
f_generated_csv = "generated/generated_mol.csv"
f_generated_txt = "generated/generated_smiles.txt"
f_data_dict_json = "generated/data.json"
f_info_json = "generated/info.json"

# parameters for loading data
#       CHON + Cl and S are allowed
atom_allowed = [16, 17]

#       not allowed in smiles
char_not_allowed = ['5', '.', '6', '7', '8', '-', '[O]', '[N+]', '[O-]', '[C]', '[N]', '[c]', '[CH]', '[CH2]', '[PH]',
                    '[S]', 'P', '[SH]', 'F', 'Br', '[As]', '[CH3]', '[OH]', '[Si]', 'B', '[SH2]', '[NH2]', '[Ge]', 'p',
                    '[P]', '[SeH]', '[Se]', '[SiH2]', '[SiH]']

# parameters for mcts
#       vocabulary, you have to update it in this file if you change the vocabulary
vocabulary = ['\n', '&', 'C', 'O', '(', '=', ')', 'c', '1', 'N', 'n',
              '2', '3', '4', '[nH]', 'Cl', 'S', 'o', '#', '[NH]', 's']

#       rates for the mcts
# the bigger the exploration rate is deeper is the tree
exploration_rate = 0.5
exporation_vs_explotation = 0.5

smiles = []
model = None
tree = None
tree_info = None
data = None
scorer = None

info_created = "nb_smiles_created"
info_good = "nb_good_smiles"
info_alrd_tested = "nb_smiles_already_tested"

# scores
logp = np.loadtxt(f_logp)
logp_mean = np.mean(logp)
logp_std = np.std(logp)
sa_scores_mean = 4
sa_scores_std = 0.5
# sa_scores = np.loadtxt(f_sa_scores)
# sa_scores_mean = np.mean(sa_scores)
# sa_scores_std = np.std(sa_scores)
cycles_scores = np.loadtxt(f_cycles_scores)
cycles_scores_mean = np.mean(cycles_scores)
cycles_scores_std = np.std(cycles_scores)

target = 500

id_smile = 0
nb_created = 0
nb_good = 0

# SCORES
s_logp = "logp"
s_sa = "sa"
s_cycle = "cycle"
s_dft = "dft"
s_id = "id"
s_valid = "valid"
