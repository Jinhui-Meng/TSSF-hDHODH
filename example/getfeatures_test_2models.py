# prepare csv file containing ID and SMILES

import os
import sys
import pandas as pd
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs import cDataStructs
from rdkit.Chem.EState.Fingerprinter import FingerprintMol


# Check mol files
def check_error(ft_ind, ft_smiles):
    for ind, smiles in zip(ft_ind, ft_smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(ind)
            break
        else:
            continue


# calculate ligand features
def generate_fingerprints(input_file_name):
    # The file should contain at least the columns "Num_ID" and "SMILES"
    # Data Load
    ft_input_df = pd.read_csv(input_file_name)

    # Using RDKit to generate three fingerprints
    ft_ECFP_id_list = []
    ft_ECFP_feature_df = pd.DataFrame()

    ft_MACCSFP_id_list = []
    ft_MACCSFP_feature_df = pd.DataFrame()

    ft_EStateFP_id_list = []
    ft_EStateFP_feature_df = pd.DataFrame()

    featurize_params = {'radius': 2,
                        'nBits': 2048,
                        'chiral': True,
                        'bonds': True,
                        'features': False,
                        'field1': 'SMILES',
                        'field2': 'Num_ID',
                        'ft_df': ft_input_df}

    ft_sample_smiles = featurize_params['ft_df'][featurize_params['field1']].tolist()
    ft_num_id = featurize_params['ft_df'][featurize_params['field2']].tolist()

    check_error(ft_num_id, ft_sample_smiles)

    for ind, smiles in zip(ft_num_id, ft_sample_smiles):
        mol = Chem.MolFromSmiles(smiles)
        # ECFP
        ECFP_feature_Bit = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,
                                                                          radius=featurize_params['radius'],
                                                                          nBits=featurize_params['nBits'],
                                                                          useChirality=featurize_params['chiral'],
                                                                          useBondTypes=featurize_params['bonds'],
                                                                          useFeatures=featurize_params['features'])
        ft_ECFP_id_list.append(ind)
        ECFP_feature = [cDataStructs.BitVectToText(ECFP_feature_Bit)]
        ft_ECFP_feature_df = pd.concat([ft_ECFP_feature_df, pd.DataFrame(ECFP_feature)], ignore_index=True)
        # MACCSFP
        MACCSFP_feature_Bit = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        ft_MACCSFP_id_list.append(ind)
        MACCSFP_feature = [cDataStructs.BitVectToText(MACCSFP_feature_Bit)]
        ft_MACCSFP_feature_df = pd.concat([ft_MACCSFP_feature_df, pd.DataFrame(MACCSFP_feature)], ignore_index=True)
        # EStateFP
        EStateFP_feature_tuple = FingerprintMol(mol)
        ft_EStateFP_id_list.append(ind)
        EStateFP_feature = pd.DataFrame(EStateFP_feature_tuple[0]).T
        ft_EStateFP_feature_df = pd.concat([ft_EStateFP_feature_df, pd.DataFrame(EStateFP_feature)], ignore_index=True)

    # ECFP
    ft_ECFP_feature_split_df = ft_ECFP_feature_df.iloc[:,0].str.split('', expand=True).drop([0,featurize_params['nBits']+1], axis = 1)
    ft_ECFP_vec_name = ['ECFP_{0}'.format(i) for i in range(1, featurize_params['nBits']+1)]
    ft_ECFP_feature_split_df.columns = ft_ECFP_vec_name
    ft_ECFP_id_df = pd.DataFrame(ft_ECFP_id_list, columns=list(['Num_ID']))
    ft_ECFP_id_feature_df = pd.merge(ft_ECFP_id_df, ft_ECFP_feature_split_df, left_index=True, right_index=True)
    print('Calculation of ECFP has been completed.')

    # MACCSFP
    MACCSFP_params = {'nBits': 167}
    ft_MACCSFP_feature_split_df = ft_MACCSFP_feature_df.iloc[:, 0].str.split('', expand=True).drop([0, MACCSFP_params['nBits']+1], axis=1)
    ft_MACCSFP_vec_name = ['MACCSFP_{0}'.format(i) for i in range(1, MACCSFP_params['nBits']+1)]
    ft_MACCSFP_feature_split_df.columns = ft_MACCSFP_vec_name
    ft_MACCSFP_id_df = pd.DataFrame(ft_MACCSFP_id_list, columns=list(['Num_ID']))
    ft_MACCSFP_id_feature_df = pd.merge(ft_MACCSFP_id_df, ft_MACCSFP_feature_split_df, left_index=True, right_index=True)
    print('Calculation of MACCSFP has been completed.')

    # EStateFP
    EStateFP_params = {'nBits': 79}
    ft_EStateFP_vec_name = ['EStateFP_{0}'.format(i) for i in range(1, EStateFP_params['nBits']+1)]
    ft_EStateFP_feature_df.columns = ft_EStateFP_vec_name
    ft_EStateFP_id_df = pd.DataFrame(ft_EStateFP_id_list, columns=list(['Num_ID']))
    ft_EStateFP_id_feature_df = pd.merge(ft_EStateFP_id_df, ft_EStateFP_feature_df, left_index=True, right_index=True)
    print('Calculation of EState has been completed.')

    # Combined three molecular fingerprints
    ft_all_id_feature_df = pd.concat([ft_ECFP_id_feature_df,
                                      ft_MACCSFP_id_feature_df.iloc[:, 1:],
                                      ft_EStateFP_id_feature_df.iloc[:, 1:]], axis=1)
    return ft_all_id_feature_df


# test ligand features selection
def LFs_select(ft_feature_df=None, selector_file=None):
    if ft_feature_df is None:
        sys.exit('No ligand features.')
    else:
        if selector_file is None:
            sys.exit('No ligand selector.')
        else:
            with open (selector_file, 'rb') as fr:
                selector_fit = pickle.load(fr)
            x_select_input = ft_feature_df.iloc[:, 1:]
            x_select_ind = ft_feature_df.iloc[:, 0]
            x_select_output = selector_fit.transform(x_select_input)
            x_all_name_list = x_select_input.columns.tolist()
            x_select_name_index = selector_fit.get_support(indices=True)
            x_select_name_list = []
            for i in x_select_name_index:
                x_select_name_list.append(x_all_name_list[i])

            x_select_df = pd.DataFrame(x_select_output)
            x_select_df.columns = x_select_name_list
            x_select_df_final = pd.merge(x_select_ind, x_select_df, left_index=True, right_index=True)

            return x_select_df_final


def estate_scale(ft_select_df=None, scaler_file=None):
    if ft_select_df is None:
        sys.exit('No estate scale input.')
    else:
        if scaler_file is None:
            sys.exit('No estate scaler.')
        else:
            with open(scaler_file, 'rb') as fr:
                Scaler_fit = pickle.load(fr)
            estate_df = ft_select_df.loc[:, ft_select_df.columns.str.contains('EStateFP')]
            estate_scaled_array = Scaler_fit.transform(estate_df)
            estate_scaled_df = pd.DataFrame(estate_scaled_array)
            ft_select_df[ft_select_df.columns[ft_select_df.columns.str.contains('EStateFP')]] = estate_scaled_df
            return ft_select_df


# test interaction features scale
def IFs_scale(features_file=None, scaler_file=None):
    if scaler_file is None:
        sys.exit('No interaction features scaler.')
    else:
        if features_file is None:
            sys.exit('No test file.')
        else:
            with open(scaler_file, 'rb') as fr:
                Scaler_fit = pickle.load(fr)
            ifs_df = pd.read_csv(features_file)
            ifs_X = np.array(ifs_df.iloc[:, 1:-1])
            ifs_X_col_name = ifs_df.iloc[:, 1:-1].columns.tolist()
            ifs_Y = ifs_df.iloc[:, 0]
            ifs_ID = ifs_df.iloc[:, -1]
            ifs_X_scaled_array = Scaler_fit.transform(ifs_X)
            ifs_X_scaled_df = pd.DataFrame(ifs_X_scaled_array)
            ifs_X_scaled_df.columns = ifs_X_col_name
            ifs_scaled_df_final = pd.concat([ifs_Y, ifs_X_scaled_df, ifs_ID], axis=1)
            print('Finished interaction features scale')
            return ifs_scaled_df_final


# Combination of scaled IFs and selected LFs
def IFsandLFs(IFs_df, LFs_df, outcwd_a, outcwd_b):
    # ligand features
    LFs_sort_df = LFs_df.sort_values(axis=0, by='Num_ID', ascending=True)
    LFs_sort_df.reset_index(drop=True)
    # every type of ligand features
    LFs_ind_select_df = LFs_sort_df.loc[:, LFs_sort_df.columns.str.contains('Num_ID')]
    LFs_ECFP_select_df = LFs_sort_df.loc[:, LFs_sort_df.columns.str.contains('ECFP')]
    # LFs_MACCSFP_select_df = LFs_sort_df.loc[:, LFs_sort_df.columns.str.contains('MACCSFP')]
    # LFs_EStateFP_select_df = LFs_sort_df.loc[:, LFs_sort_df.columns.str.contains('EStateFP')]
    
    # interaction features
    IFs_sort_df = IFs_df.sort_values(axis=0, by='PDB', ascending=True)
    IFs_sort_df.reset_index(drop=True)
    LFs_xy_scaled_df = IFs_sort_df.iloc[:, 0:-1]
    LFs_ind_scaled_df = IFs_sort_df.iloc[:, -1]
    
    # Features combination
    # TSSF-A (IFs+ECFP)
    allft_tssfa = pd.concat([LFs_xy_scaled_df, LFs_ECFP_select_df, LFs_ind_scaled_df], axis=1)
    # TSSF-B (IFs+LFs)
    allft_tssfb = pd.concat([LFs_xy_scaled_df, LFs_sort_df.iloc[:, 1:], LFs_ind_scaled_df], axis=1)

    # output csv
    allft_tssfa.to_csv(outcwd_a, sep=",", index=None)
    print('Output features file TSSF-a:', outcwd_a)
    allft_tssfb.to_csv(outcwd_b, sep=",", index=None)
    print('Output features file TSSF-b:', outcwd_b)

    
    
if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.join(os.getcwd(), "..")))
    selector_fit = './features/train_features_selector.pkl'
    estate_scaler = './features/EstateFP_scaler.pkl'
    IFs_scaler = './features/IFs_scaler.pkl'
    IFs_raw = './example/data/data_yx42i.csv'
    smiles_inf = './example/data/data_smiles.csv'
    outa_path = './example/test_features_a.csv'
    outb_path = './example/test_features_b.csv'

    # features generation
    LFs_raw_df = generate_fingerprints(smiles_inf)
    LFs_select_df = LFs_select(ft_feature_df=LFs_raw_df, selector_file=selector_fit)
    LFs_scale_df = estate_scale(ft_select_df=LFs_select_df, scaler_file=estate_scaler)
    IFs_scale_df = IFs_scale(features_file=IFs_raw, scaler_file=IFs_scaler)
    IFsandLFs(IFs_df=IFs_scale_df, LFs_df=LFs_scale_df, outcwd_a=outa_path, outcwd_b=outb_path)

