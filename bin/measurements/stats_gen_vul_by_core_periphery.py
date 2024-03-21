import numpy as np
import pandas as pd

import argparse
from configparser import ConfigParser
from ..utils.file_handlers import get_dataset, progression_bar, save_df_to_pickle
from ..utils.core_periphery import *

# ArgParser
parser = argparse.ArgumentParser(description = "Create similarity dataset")
parser.add_argument("--config", "-c", type = str, help = "path/to/config.ini", required=True)
args = parser.parse_args()
config_path = args.config

# Reading settings
config = ConfigParser()
config.read(config_path)

dataset_path = config.get('dataset', 'dataset_path')
gen_vul_df_path = config.get('core and periphery', 'gen_vul_df_path')
gen_vul_composition_df_path = config.get('core and periphery', 'gen_vul_composition_df_path')

G_dataset = get_dataset(dataset_path, verbose = False)
gen_vul_df = pd.read_pickle(gen_vul_df_path)
dataset_len = len(G_dataset)
rows = []
for graph_index, G in enumerate(G_dataset):

    graph_name = G['name']
    V = G.vcount()
    df_temp_liv = gen_vul_df[(gen_vul_df['Network'] == G['name']) & (gen_vul_df['ECO'] == 1)]

    n_gen = get_n_gen(df_temp_liv)
    n_vul = get_n_vul(df_temp_liv)
    n_gen_and_vul = get_n_gen_and_vul(df_temp_liv)

    df_temp_liv_core = df_temp_liv[df_temp_liv['Structure'] == 'Core']
    V_core = df_temp_liv_core.shape[0]
    if V_core > 0:
        n_gen_core = get_n_gen(df_temp_liv_core)
        n_vul_core = get_n_vul(df_temp_liv_core)
        n_gen_and_vul_core = get_n_gen_and_vul(df_temp_liv_core)
        p_gen_core = n_gen_core / V_core
        p_vul_core = n_vul_core / V_core
        p_gen_and_vul_core = n_gen_and_vul_core / V_core
        p_all_gen_core = n_gen_core / n_gen
        p_all_vul_core = n_vul_core / n_vul
    else:
        p_gen_core = None
        p_vul_core = None
        p_gen_and_vul_core = None
        p_all_gen_core = None
        p_all_vul_core = None

    df_temp_liv_periphery = df_temp_liv[df_temp_liv['Structure'] != 'Core']
    V_periphery = df_temp_liv_periphery.shape[0]
    if V_periphery > 0:
        n_gen_periphery = get_n_gen(df_temp_liv_periphery)
        n_vul_periphery = get_n_vul(df_temp_liv_periphery)
        n_gen_and_vul_periphery = get_n_gen_and_vul(df_temp_liv_periphery)
        p_gen_periphery = n_gen_periphery / V_periphery
        p_vul_periphery = n_vul_periphery / V_periphery
        p_gen_and_vul_periphery = n_gen_and_vul_periphery / V_periphery
        p_all_gen_periphery = n_gen_periphery / n_gen
        p_all_vul_periphery = n_vul_periphery / n_vul
    else:
        p_gen_periphery = None
        p_vul_periphery = None
        p_gen_and_vul_periphery = None
        p_all_gen_periphery = None
        p_all_vul_periphery = None

    row = (G['name'], V,
        n_gen / V, n_vul / V, n_gen_and_vul / V,
        p_gen_core, p_vul_core, p_gen_and_vul_core, p_all_gen_core, p_all_vul_core,
        p_gen_periphery, p_vul_periphery, p_gen_and_vul_periphery,
        p_all_gen_periphery, p_all_vul_periphery)
    rows.append(row)
    progression_bar(graph_index +1, dataset_len)

print()

gen_vul_composition_df = pd.DataFrame(rows, columns = ['Network', 'Nr nodes',
                              '% gen network', '% vul network', '% gen and vul network',
                              '% gen core', '% vul core', '% gen and vul core', '% all gen in core', '% all vul in core',
                              '% gen periphery', '% vul periphery', '% gen and vul periphery',
                              '% all gen in periphery', '% all vul in periphery'])

save_df_to_pickle(gen_vul_composition_df, gen_vul_composition_df_path)