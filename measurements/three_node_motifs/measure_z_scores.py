import sys  
sys.path.append('/home/davide/AI/Projects/AquaNet') # Initialize Experiment

import numpy as np
import pandas as pd
from utils.data_handling import load_dataset, print_time, print_progress, create_folder
from utils.preprocessing import get_subgraph_living_creature
from utils.measurements import motif_contained_in_adj_fast, process_randomization, count_first_order_motifs, calculate_z_score_from_experiments, swap_adj_matrix
from settings import verbose, dataset_folder, num_random_graph_samples_triad, df_motif_Z_scores_path, df_motif_randomization_data_path, seed
import concurrent.futures
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import igraph as ig

def main():
    G_dataset = load_dataset(dataset_folder)
    random_df_list = []
    z_score_df_list = []
    create_folder(df_motif_Z_scores_path, verbose = verbose)
    create_folder(df_motif_randomization_data_path, verbose = verbose)
    if seed is not None: 
        np.random.seed(seed)

    if verbose:
        print_time("Start")
        start_time = datetime.now().timestamp()
    seed_table = [np.random.randint(0, 2**32) for i in range(num_random_graph_samples_triad)]
    
    for i_G, G in enumerate(G_dataset):
        if verbose:
            print_progress("Randomization and Z-score calculation", i_G, len(G_dataset))
        G_liv = get_subgraph_living_creature(G)
        rows_G_liv = []
        adjacency_matrix = np.array(G_liv.get_adjacency().data)
        motif_census = motif_contained_in_adj_fast(adjacency_matrix)
        single_edges, double_edges = count_first_order_motifs(adjacency_matrix)
        rows_G_liv.append(['Original', G['name'], motif_census, single_edges, double_edges])

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_randomization, adjacency_matrix, adj_rand_function = swap_adj_matrix, seed = seed_table[i]) for i in range(num_random_graph_samples_triad)]

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            motif_census_rand, single_edges, double_edges = future.result()[:3]
            rows_G_liv.append([f'Rand_{i:03}', G['name'], motif_census_rand, single_edges, double_edges])
        random_df_G = pd.DataFrame(rows_G_liv, columns = ['ID', 'Network', 'Triad Census', 'Single', 'Double'])
        random_df_list.append(random_df_G)
        z_score, motif_order = calculate_z_score_from_experiments(random_df_G)
        z_score_graph = {motif: z_score[i] for i, motif in enumerate(motif_order)}
        z_score_graph['Network'] = G['name']
        z_score_df_list.append(pd.DataFrame(z_score_graph, index = [i_G]))

        random_df = pd.concat(random_df_list)
        random_df.to_pickle(df_motif_randomization_data_path)
        z_score_df = pd.concat(z_score_df_list)
        z_score_df.to_pickle(df_motif_Z_scores_path)

    if verbose:
        print_progress("Randomization and Z-score calculation", len(G_dataset), len(G_dataset))
        print()
        print_time(f"Finished in {datetime.now().timestamp() - start_time:.2f} seconds")
    random_df = pd.concat(random_df_list)
    random_df.to_pickle(df_motif_randomization_data_path)
    if verbose:
        print_time(f"Randomization dataframe saved to {df_motif_randomization_data_path}")
    
    z_score_df = pd.concat(z_score_df_list)
    z_score_df.to_pickle(df_motif_Z_scores_path)
    if verbose:
        print_time(f"Z-score dataframe saved to {df_motif_Z_scores_path}")

    return

if __name__ == '__main__':
    main()