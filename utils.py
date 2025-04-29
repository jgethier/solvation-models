
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors

font_names = [f.name for f in fm.fontManager.ttflist]

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2
plt.style.use('tableau-colorblind10')


def get_figure(figsize=(5.5,4)):

    fig, ax = plt.subplots(figsize=figsize)

    ax.xaxis.set_tick_params(which='major', size=7, width=2, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=4, width=2, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=7, width=2, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=4, width=2, direction='in', right='on')

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax.tick_params(direction='in')

    return fig, ax 


def generate_descriptors(smiles):
    """
    Generates molecular descriptors for a given SMILES string.

    Args:
    smiles: A SMILES string representing a molecule.

    Returns:
    A dictionary of molecular descriptors, where the keys are descriptor names
    and the values are the corresponding descriptor values.
    """


    #smiles = standardize_smiles(smiles)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(smiles)
        return {}

    descriptors = {}
    for descriptor_name, descriptor_function in Descriptors.descList:
        try:
            descriptors[descriptor_name] = descriptor_function(mol)
        except:
            descriptors[descriptor_name] = None  # Set to None if calculation fails

    return descriptors


def prepare_descriptors(df,label_column,theory_column=None):

    solute_descriptors = []

    for smiles_str in df['Solute SMILES']:
        descr = generate_descriptors(smiles_str)
        solute_descriptors.append(descr)

    df_descr_solute = pd.DataFrame(solute_descriptors)
    df_descr_solute['Solute SMILES'] = df['Solute SMILES']

    solvent_descriptors = []

    for smiles_str in df['Solvent SMILES']:
        descr = generate_descriptors(smiles_str)
        solvent_descriptors.append(descr)


    df_descr_solvent = pd.DataFrame(solvent_descriptors)
    df_descr_solvent['Solvent SMILES'] = df['Solvent SMILES']

    df_descr_solute = df_descr_solute.add_suffix('_solute')
    df_descr_solvent = df_descr_solvent.add_suffix('_solvent')

    total_df = pd.concat([df_descr_solute,df_descr_solvent],axis=1)

    if theory_column is not None:
        total_df['Calculated_deltaG'] = df[theory_column]

    total_df['Experimental_deltaG'] = df[label_column]

    return total_df

def bin_rotors_above_ten(row):

    if row['Number of rotors (non-H)'] >= 10:
        return 10
    else:
        return row['Number of rotors (non-H)']
