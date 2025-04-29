import pandas as pd
import numpy as np

import utils


def load_minnesota_data(theory_column='DeltaGsolv uESE (kcal/mol) - 1'):

    df = pd.read_csv('./minnesota_all_data_solvation_fe_corrected.csv')

    minn_df = utils.prepare_descriptors(df,label_column='DeltaGsolv (kcal/mol)',theory_column=theory_column)

    nunique = minn_df.nunique()
    cols_to_drop = nunique[nunique == 1].index
    minn_df = minn_df.drop(cols_to_drop,axis=1)

    minn_columns = minn_df.columns
    solute = 0
    solvent = 0
    for j in range(0,len(minn_columns)):
        if 'solute' in minn_columns[j]:
            solute+=1
        if 'solvent' in minn_columns[j]:
            solvent+=1
    print("Total Solute Descriptors: %d"%solute)
    print("Total Solvent Descriptors: %d"%solvent)

    nan_rows = minn_df.isna().any(axis=1)
    inf_rows = minn_df[(minn_df == np.inf).any(axis=1)].index
    print("Number of rows with at least 1 NaN value:", len(nan_rows[nan_rows==True]))
    print("Number of rows with at least 1 infinity value:", len(inf_rows))

    index_values = minn_df[nan_rows].index.tolist()
    minn_df = minn_df.drop(index_values)

    df_copy = df.copy().drop(index_values)

    df_copy['Stratify_by_rotors'] = df_copy.apply(lambda row: utils.bin_rotors_above_ten(row), axis=1)

    return minn_df, df_copy


def load_dGsolv_data(columns,theory_column='dGsolv uESE [kcal/mol] - 1'):

    dGsolv_DB_df = pd.read_csv('dGsolvDB1_all_data_solvation_fe_fixed.csv')

    dGsolv_df = utils.prepare_descriptors(dGsolv_DB_df,label_column='dGsolv_avg [kcal/mol]',theory_column=theory_column)

    dGsolv_df = dGsolv_df[columns]

    solute = 0
    solvent = 0
    for j in range(0,len(columns)):
        if 'solute' in columns[j]:
            solute+=1
        if 'solvent' in columns[j]:
            solvent+=1
    print("Total Solute Descriptors: %d"%solute)
    print("Total Solvent Descriptors: %d"%solvent)

    nan_rows = dGsolv_df.isna().any(axis=1)
    inf_rows = dGsolv_df[(dGsolv_df == np.inf).any(axis=1)]
    print("Number of rows with at least 1 NaN value:", len(nan_rows[nan_rows==True]))
    print("Number of rows with at least 1 infinity value:", len(inf_rows))

    index_values = dGsolv_df[nan_rows].index.tolist()
    dGsolv_df = dGsolv_df.drop(index_values)
    dGsolv_df_copy = dGsolv_DB_df.copy().drop(index_values)

    dGsolv_df_copy['Stratify_by_rotors'] = dGsolv_df_copy.apply(lambda row: utils.bin_rotors_above_ten(row), axis=1)

    return dGsolv_df, dGsolv_df_copy


def load_combined_data():

    df = pd.read_csv('./minnesota_all_data_solvation_fe_corrected.csv')
    minn_df = utils.prepare_descriptors(df,label_column='DeltaGsolv (kcal/mol)',theory_column='DeltaGsolv uESE (kcal/mol) - 1')

    dGsolv_DB_df = pd.read_csv('dGsolvDB1_all_data_solvation_fe_fixed.csv')
    dGsolv_df = utils.prepare_descriptors(dGsolv_DB_df,label_column='dGsolv_avg [kcal/mol]',theory_column='dGsolv uESE [kcal/mol] - 1')

    combined_df = pd.concat([minn_df, dGsolv_df],ignore_index=True)

    nunique = combined_df.nunique()
    cols_to_drop = nunique[nunique == 1].index
    combined_df = combined_df.drop(cols_to_drop,axis=1)

    minn_columns = combined_df.columns
    solute = 0
    solvent = 0
    for j in range(0,len(minn_columns)):
        if 'solute' in minn_columns[j]:
            solute+=1
        if 'solvent' in minn_columns[j]:
            solvent+=1
    print("Total Solute Descriptors: %d"%solute)
    print("Total Solvent Descriptors: %d"%solvent)

    nan_rows = combined_df.isna().any(axis=1)
    inf_rows = combined_df[(combined_df == np.inf).any(axis=1)].index
    #duplicate_rows = combined_df[(combined_df.duplicated(subset=['Solvent SMILES_solvent','Solute SMILES_solute'])==True)].index
    print("Number of rows with at least 1 NaN value:", len(nan_rows[nan_rows==True]))
    print("Number of rows with at least 1 infinity value:", len(inf_rows))
    #print("Number of duplicated solute/solvent pairs:", len(duplicate_rows),duplicate_rows.tolist())
    #print(combined_df[['Solute SMILES_solute','Solvent SMILES_solvent']].iloc(duplicate_rows.tolist()))

    index_values = combined_df[nan_rows].index.tolist()
    combined_df = combined_df.drop(index=index_values).reset_index(drop=True)
    
    return combined_df


if __name__=='__main__':
    load_combined_data()
