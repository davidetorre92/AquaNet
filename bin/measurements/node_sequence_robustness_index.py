import numpy as np
import pandas as pd

import os
import argparse
from configparser import ConfigParser
from ..utils.file_handlers import get_dataset, progression_bar, save_df_to_pickle
from ..utils.connectiveness import *

# ArgParser
parser = argparse.ArgumentParser(description = "Create similarity dataset")
parser.add_argument("--config", "-c", type = str, help = "path/to/config.ini", required=True)
args = parser.parse_args()
config_path = args.config

# Reading settings
config = ConfigParser()
config.read(config_path)

dataset_path = config.get('dataset', 'dataset_path')
df_node_sequence_path = config.get('critical nodes', 'df_node_sequence_path')
df_robustness_path = config.get('critical nodes', 'df_robustness_path')

G_dataset = get_dataset(dataset_path, verbose = False)

df_node_removal = []
df_robustness = []
dataset_len = len(G_dataset)

for index_graph, G in enumerate(G_dataset):
    graph_name = G['name']
    list_of_removed_nodes = node_removal_strategy(G, measure_v = lambda G, v: greedy_G_v(G, v, n_reachable_pairs))
    df_node_removal_G = get_reachable_pairs_change(G, list_of_removed_nodes)
    rho_G = robustness_function_reachable_pairs(df_node_removal_G)
    df_robustness.append((graph_name, rho_G))
    df_node_removal.append(df_node_removal_G)
    progression_bar(index_graph +1, dataset_len)

print()
df_node_removal = pd.concat(df_node_removal)
df_robustness = pd.DataFrame(df_robustness, columns = ['graph_name', 'rho_G'])
df_node_removal['node_fraction'] = df_node_removal['node_fraction'].map(lambda x: x*100)

save_df_to_pickle(df_node_removal, df_node_sequence_path)
save_df_to_pickle(df_robustness, df_robustness_path)



