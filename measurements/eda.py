import sys  
sys.path.append('/home/davide/AI/Projects/AquaNet') # Initialize Experiment

import numpy as np
import pandas as pd
from utils.preprocessing import get_subgraph_living_creature
from utils.data_handling import load_dataset, print_progress, print_time, create_folder
from settings import df_eda_path, dataset_folder, verbose

def main():
    create_folder(df_eda_path, verbose = verbose)
    G_dataset = load_dataset(dataset_folder)
    eda_df = {'Network': [], 
              'S': [], 
              'L': [], 
              'C': [], 
              'Det': [], 
              'Basal': [], 
              'Giant Strongly Connected Component': [], 
              'Max Trophic Level': []}
    for i_G, G in enumerate(G_dataset):
        if verbose: print_progress("Analyzing graph", i_G + 1, len(G_dataset))
        S = G.vcount()
        L = G.ecount()
        connectance_G_all = L / (S * S)
        det = len([v for v in G.vs() if v['ECO'] == 2])
        basal = len([v for v in G.vs() if v.indegree() == 0])
        components = G.components(mode='strong')
        giant_scc = max(components, key=len)
        giant_scc_fraction = len(giant_scc) / G.vcount()
        max_trophic_level = max(G.vs['trophic_level'])
        eda_df['Network'].append(G['name'])
        eda_df['S'].append(G.vcount())
        eda_df['L'].append(G.ecount())
        eda_df['C'].append(connectance_G_all)
        eda_df['Det'].append(det)
        eda_df['Basal'].append(basal)
        eda_df['Giant Strongly Connected Component'].append(giant_scc_fraction)
        eda_df['Max Trophic Level'].append(max_trophic_level)
    if verbose: 
        print_progress("Analyzing graph", i_G + 1, len(G_dataset))
        print()
    eda_df = pd.DataFrame(eda_df)
    eda_df.to_pickle(df_eda_path)
    if verbose: print_time(f"EDA analysis complete. File saved in {df_eda_path}")
    # Display Mean, Std Dev, Min and Max for each parameter
    for col in eda_df.columns:
        if col == 'Network': continue
        print(col)
        print(f"\tMean: {np.mean(eda_df[col])}")
        print(f"\tStd Dev: {np.std(eda_df[col])}")
        print(f"\tMin: {np.min(eda_df[col])}")
        print(f"\tMax: {np.max(eda_df[col])}")
if __name__ == '__main__':
    main()