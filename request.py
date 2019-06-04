import json
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

import parameters as p
from plot_wavelength import plot_wl


def select(starting_with='', wl_min=0, wl_max=float('inf'), unit="nm", f_min=0.0):
    selected_smiles = set()
    for smiles in p.data.keys():
        if p.data[smiles]['valid']:
            if smiles.startswith(starting_with):
                for line in p.data[smiles][p.s_dft]:
                    if wl_min <= line[unit] <= wl_max and line['f'] >= f_min:
                        selected_smiles.add(smiles)
                        print(smiles)
                        print(line)

    return selected_smiles


def smiles_to_image(id, smiles):
    m = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(m)
    Draw.MolToFile(m, str(id) + '_2D.png')
    # Draw.MolToFile(m, "generated/dft/" + str(id) + '.png')


if __name__ == '__main__':
    with open(p.f_data_dict_json) as d:  # 'data.json') as d:  #
        p.data = json.load(d)

    selected = select(starting_with="", wl_min=500, unit="nm", f_min=0.1)

    for s in selected:
        if p.btx:
            smi = "".join(p.prefix) + s[1:-1]
        else:
            smi = s[1:-1]
        smiles_to_image(p.data[s]['id'], smi)
        plot_wl(s)
