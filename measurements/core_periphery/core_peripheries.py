import sys  
sys.path.append('/home/davide/AI/Projects/AquaNet') # Initialize Experiment

import numpy as np
import pandas as pd

from utils.data_handling import load_dataset, print_time, print_progress, create_folder
from utils.measurements import get_core_periphery_structure
from settings import dataset_folder, verbose, df_nodes_core_periphery_classification_path, df_links_core_periphery_classification_path

def main():
    G_dataset = load_dataset(dataset_folder)
    nodes_df = []
    links_df = []
    key_v = lambda v: v['name']
    for i_G, G in enumerate(G_dataset):
        if verbose:
            print_progress("Evaluating core - periphery structure", i_G, len(G_dataset))   
        core_periphery_to_index_dict = get_core_periphery_structure(G)
        index_to_core_periphery_dict = {}
        for key, indices_set in core_periphery_to_index_dict.items():
            for index in indices_set:
                index_to_core_periphery_dict[index] = key
                G.vs[index]['core_periphery'] = key

        links_df_G = pd.DataFrame([(
            G['name'],
            key_v(G.vs[e.source]),
            key_v(G.vs[e.target]),
            G.vs[e.source]['core_periphery'],
            G.vs[e.target]['core_periphery'],
            G.vs[e.source]['ECO'],
            G.vs[e.target]['ECO'],
            G.vs[e.source]['trophic_level'],
            G.vs[e.target]['trophic_level'],
            ) for e in G.es()], columns = ['Network', 'Source name', 'Target name', 'Source core periphery', 'Target core periphery', 'Source ECO', 'Target ECO', 'Source TL', 'Target TL'])
        nodes_df_G = pd.DataFrame([[G['name'], key_v(v), v['core_periphery'], v['ECO'], v['trophic_level']] for v in G.vs], columns = ['Network', 'Node name', 'Core periphery', 'ECO', 'Trophic level'])
                
        nodes_df.append(nodes_df_G)
        links_df.append(links_df_G)

    if verbose:
        print_progress("Evaluating core - periphery structure", len(G_dataset), len(G_dataset))   
        print()
    nodes_df = pd.concat(nodes_df)
    links_df = pd.concat(links_df)
    create_folder(df_nodes_core_periphery_classification_path, verbose = True)
    create_folder(df_links_core_periphery_classification_path, verbose = True)
    nodes_df.to_pickle(df_nodes_core_periphery_classification_path)
    links_df.to_pickle(df_links_core_periphery_classification_path)
    if verbose:
        print_time(f'Nodes classification dataframe saved in {df_nodes_core_periphery_classification_path}')
        print_time(f'Links classification dataframe saved in {df_links_core_periphery_classification_path}')

if __name__ == '__main__':
    main()
