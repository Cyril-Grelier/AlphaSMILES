from __future__ import absolute_import, division, print_function

import csv
from builtins import (open, range, int)

"""
Structure of a line :
row :
 0   'id',
 1   'SMILES',
 2   'energies of electronic transitions 0',
 3   'energies of electronic transitions 1',
 4   'energies of electronic transitions 2',
 5   'energies of electronic transitions 3',
 6   'energies of electronic transitions 4',
 7   'energies of electronic transitions 5',
 8   'energies of electronic transitions 6',
 9   'energies of electronic transitions 7',
 10  'energies of electronic transitions 8',
 11   'energies of electronic transitions 9',
 12   'oscillator strengths of transitions 0',
 13   'oscillator strengths of transitions 1',
 14   'oscillator strengths of transitions 2',
 15   'oscillator strengths of transitions 3',
 16   'oscillator strengths of transitions 4',
 17   'oscillator strengths of transitions 5',
 18   'oscillator strengths of transitions 6',
 19   'oscillator strengths of transitions 7',
 20   'oscillator strengths of transitions 8',
 21   'oscillator strengths of transitions 9',
 22   'logP',
 23   'SA Score',
 24   'cycle score',
 25   'molecules \\{CHON}',
 26   'only {CHON}'
"""


def is_row_ok(row, atom_allowed=None, char_not_allowed=None):
    if char_not_allowed is None:
        char_not_allowed = []
    if atom_allowed is None:
        atom_allowed = []
    if not atom_allowed:
        if row[-1] == "0":
            # if only CHON is allowed and the last line is 0 then return False
            return False
    else:
        # if there is only some atoms allowed
        if row[-2]:
            list_atom = map(int, row[-2].split(":"))
            for allowed in atom_allowed:
                if allowed not in list_atom:
                    return False

    if char_not_allowed:
        for c in char_not_allowed:
            if c in row[1]:
                return False
    return True


def load_dict(csvfile="data/data.csv", atom_allowed=None, char_not_allowed=None):
    if char_not_allowed is None:
        char_not_allowed = []
    if atom_allowed is None:
        atom_allowed = []
    with open(csvfile, 'r') as csvf:
        all_data = csv.reader(csvf, delimiter=',')
        data = dict()
        for row in all_data:
            try:
                if is_row_ok(row, atom_allowed=atom_allowed, char_not_allowed=char_not_allowed):
                    dft = []
                    for i in range(2, 12):
                        dft.append((float(row[i]), float(row[i + 10])))
                    data[row[1]] = {"id": int(row[0]),
                                    "logp": float(row[22]),
                                    "sa": float(row[23]),
                                    "cycle": float(row[24]),
                                    "dft": dft}

            except Exception as e:
                print(e)
                print(row[0])
    return data


def convertion_nm_to_cm_m1(val):
    """
    http://halas.rice.edu/conversions
    100 nm = 100000.00000
    150 nm = 66666.66667
    200 nm = 50000.00000
    250 nm = 40000.00000
    300 nm = 33333.33333
    350 nm = 28571.42857
    400 nm = 25000.00000
    450 nm = 22222.22222
    500 nm = 20000.00000
    550 nm = 18181.81818
    600 nm = 16666.66667
    650 nm = 15384.61538

    x nm = 10,000,000 / x cm-1
    y cm-1 = 10,000,000 / y nm

    """
    return 10000000 / val


if __name__ == '__main__':
    atom_allowe = [16, 17]
    char_not_allowe = ['5', '.', '6', '7', '8', '-', '[O]', '[N+]', '[O-]', '[C]', '[N]', '[c]', '[CH]', '[CH2]', '[PH]',
                    '[S]', 'P', '[SH]', 'F', 'Br', '[As]', '[CH3]', '[OH]', '[Si]', 'B', '[SH2]', '[NH2]', '[Ge]', 'p',
                    '[P]', '[SeH]', '[Se]', '[SiH2]', '[SiH]', 'C', 'N']
    dat = load_dict(atom_allowed=atom_allowe, char_not_allowed=char_not_allowe)
    from pprint import pprint
    pprint(len(dat))
