import sys  
sys.path.append('/home/davide/AI/Projects/AquaNet') # Initialize Experiment

import numpy as np
import pandas as pd
from utils.data_handling import load_dataset, print_time, print_progress
from utils.measurements import *
from settings import dataset_folder, df_critical_nodes_path, df_robustness_path, verbose

def main():

    G_dataset = load_dataset(dataset_folder)
    robustness_df = {'Network': [], 'rho_G': []}
    critical_nodes_df = pd.read_pickle(df_critical_nodes_path)
    if verbose: print_time("Start")
    for i_G, G in enumerate(G_dataset):
        if verbose: print_progress("Critical nodes sequence", i_G + 1, len(G_dataset), n_char = 100)
        critical_nodes_data = critical_nodes_df[critical_nodes_df['Network'] == G['name']][['name', 'seq_index']].to_dict()
        # If dataset is empty print network name and continue
        if len(critical_nodes_data['name']) == 0:
            print(f"{G['name']} not available. Skipping...")
        # Check critical_nodes_indices correcteness
        if sorted(critical_nodes_data['name'].keys()) != sorted(critical_nodes_data['seq_index'].keys()):
            raise ValueError(f"Critical nodes indices not correct in the following network: {G['name']}")
        else:
            critical_nodes_indices = sorted(critical_nodes_data['name'].keys())
        node_seq_index_tuple = [(critical_nodes_data['name'][i], critical_nodes_data['seq_index'][i]) for i in critical_nodes_indices]
        # Sort node_seq_index_tuple by seq_index
        node_seq_index_tuple.sort(key = lambda x: x[1])
        critical_nodes_names = [x[0] for x in node_seq_index_tuple]
        robustness_of_the_graph = rho_G(G, critical_nodes_names)
        robustness_df['Network'].append(G['name'])
        robustness_df['rho_G'].append(robustness_of_the_graph)
        pd.DataFrame(robustness_df).to_pickle(df_robustness_path)
    if verbose:
        print_progress("Critical nodes sequence", len(G_dataset), len(G_dataset), n_char = 100)
        print()
    if verbose:
        print_time(f"Saved robustness in {df_robustness_path}")
        print_time("End")
if __name__ == "__main__":
    main()