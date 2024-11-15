import sys  
sys.path.append('/home/davide/AI/Projects/AquaNet') # Initialize Experiment

import numpy as np
import pandas as pd
from utils.data_handling import print_time, print_progress, load_dataset
from settings import verbose, dataset_folder, df_critical_nodes_path, df_kendalltau_path
from scipy.stats import kendalltau
def main():
    G_dataset = load_dataset(dataset_folder)
    critical_nodes_df = pd.read_pickle(df_critical_nodes_path)
    df_kendalltau = {'Network': [], '$CSeq_G$ / ind': [], '$CSeq_G$ / out': [], '$CSeq_G$ / ind + out': []}
    if verbose: print_time("Start")
    for i_G, G in enumerate(G_dataset):
        if verbose: 
            print_progress("Critical Node Degree", i_G + 1, len(G_dataset))
        # Get critical nodes data
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
        # Get critical nodes indegree and sort
        indegree_tuple = [(v['name'],v.indegree()) for v in G.vs if v['name'] in critical_nodes_names]
        indegree_tuple.sort(key = lambda x: x[1], reverse = True)
        critical_nodes_indegree = [v[0] for v in indegree_tuple]
        # Get critical nodes outdegree
        outdegree_tuple = [(v['name'],v.outdegree()) for v in G.vs if v['name'] in critical_nodes_names]
        outdegree_tuple.sort(key = lambda x: x[1], reverse = True)
        critical_nodes_outdegree = [v[0] for v in outdegree_tuple]
        # Get critical nodes indegree + outdegree
        in_plus_out_tuple = [(v['name'], v.indegree() + v.outdegree()) for v in G.vs if v['name'] in critical_nodes_names]
        in_plus_out_tuple.sort(key = lambda x: x[1], reverse = True)
        critical_nodes_in_plus_out = [v[0] for v in in_plus_out_tuple]
        # Compute Kendall's tau
        tau_cseq_ind, _ = kendalltau(critical_nodes_indegree, critical_nodes_names)
        tau_cseq_outd, _ = kendalltau(critical_nodes_outdegree, critical_nodes_names)
        tau_cseq_ind_plus_out, _ = kendalltau(critical_nodes_in_plus_out, critical_nodes_names)
        # Save
        df_kendalltau['Network'].append(G['name'])
        df_kendalltau['$CSeq_G$ / ind'].append(tau_cseq_ind)
        df_kendalltau['$CSeq_G$ / out'].append(tau_cseq_outd)
        df_kendalltau['$CSeq_G$ / ind + out'].append(tau_cseq_ind_plus_out)
    if verbose: 
        print_progress("Critical Node Degree", len(G_dataset), len(G_dataset))
        print()
        print_time("Done")
    df_kendalltau = pd.DataFrame(df_kendalltau)
    df_kendalltau.to_pickle(df_kendalltau_path)
    print()
    print(df_kendalltau.mean(numeric_only=True))
    if verbose: print_time(f"Saved Kendall's tau in {df_kendalltau_path}")
if __name__ == "__main__":
    main()