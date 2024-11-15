import sys
sys.path.append('/home/davide/AI/Projects/AquaNet') # Initialize Experiment

import numpy as np
import pandas as pd
from utils.data_handling import load_dataset, print_time, print_progress
from utils.preprocessing import get_subgraph_living_creature
from settings import dataset_folder, df_critical_nodes_path, df_robustness_path, verbose

N = 5
df_critical_nodes = pd.read_pickle(df_critical_nodes_path)
df_robustness = pd.read_pickle(df_robustness_path)

G_dataset = load_dataset(dataset_folder)
df_rho_std = {'Network': [], 'rho_G': [], f'std_top_{N}': []}
if verbose: print_time("Start")
for i_G, G in enumerate(G_dataset):
    if verbose: print_progress("Critical nodes sequence", i_G + 1, len(G_dataset), n_char = 100)
    critical_nodes_data = df_critical_nodes[df_critical_nodes['Network'] == G['name']][['name', 'seq_index']]
    # If dataset is empty print network name and continue
    if critical_nodes_data.shape[0] == 0:
        print(f"{G['name']} not available. Skipping...")
    G_liv = get_subgraph_living_creature(G)
    trophic_level_G = np.array([v['trophic_level'] for v in G_liv.vs])
    rho_G = df_robustness[df_robustness['Network'] == G['name']]['rho_G'].iloc[0]
    top_N_critical_nodes = [critical_nodes_data['name'][i] for i in range(N)]
    normalized_trophic_level_G = (trophic_level_G - min(trophic_level_G))/(max(trophic_level_G) - min(trophic_level_G))
    top_N_critical_nodes_trophic_level = [normalized_trophic_level_G[v.index] for v in G_liv.vs if v['name'] in top_N_critical_nodes]
    df_rho_std['Network'].append(G['name'])
    df_rho_std['rho_G'].append(rho_G)
    df_rho_std[f'std_top_{N}'].append(np.std(top_N_critical_nodes_trophic_level))

if verbose:
    print_progress("Critical nodes sequence", len(G_dataset), len(G_dataset), n_char = 100)
    print()

df_rho_std = pd.DataFrame(df_rho_std)
print(df_rho_std.corr(numeric_only = True))