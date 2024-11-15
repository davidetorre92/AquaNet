import sys  
sys.path.append('/home/davide/AI/Projects/AquaNet') # Initialize Experiment

import numpy as np
import pandas as pd
from utils.data_handling import load_dataset, print_time, print_progress, create_folder
from utils.measurements import *
from settings import dataset_folder, df_critical_nodes_path, verbose

def main():

    G_dataset = load_dataset(dataset_folder)
    critical_nodes_dfs = []
    attributes = {'name': lambda v: v['name'], 'trophic_level': lambda v: v['trophic_level'], 'ECO': lambda v: v['ECO']}
    create_folder(df_critical_nodes_path, verbose = verbose)
    if verbose: print_time("Start")
    for i_G, G in enumerate(G_dataset):
        critical_nodes_names = get_critical_nodes_sequence(G,
                                                            measure_v = lambda G, v: remove_v_function(G, v, function = n_reachable_pairs), 
                                                            vertices_ordering_function = lambda nodes,
                                                            f_G_del: min_index_attr(nodes, f_G_del, reverse = False)
                                                        )
        names_to_attributes = {v['name']: {key: attribute(v) for key, attribute in attributes.items()} for v in G.vs}        
        if verbose: print_progress("Critical nodes sequence", i_G+1, len(G_dataset), n_char = 100)
        df_G = pd.DataFrame([[names_to_attributes[name][attribute] for attribute in attributes.keys()] for name in critical_nodes_names], columns = attributes.keys())
        df_G['Network'] = G['name']
        df_G['seq_index'] = [i+1 for i in range(df_G.shape[0])]
        critical_nodes_dfs.append(df_G)
        critical_nodes_df = pd.concat(critical_nodes_dfs)
        critical_nodes_df.to_pickle(df_critical_nodes_path)
    if verbose:
        print_progress("Critical nodes sequence", len(G_dataset), len(G_dataset), n_char = 100)
        print()
    if verbose:
        print_time(f"Saved critical nodes in {df_critical_nodes_path}")
        print_time("End")
if __name__ == "__main__":
    main()