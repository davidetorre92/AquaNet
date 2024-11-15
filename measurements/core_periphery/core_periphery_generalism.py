import sys  
sys.path.append('/home/davide/AI/Projects/AquaNet') # Initialize Experiment

from datetime import datetime
import numpy as np
import pandas as pd
from utils.data_handling import load_dataset, print_progress, print_time
from utils.preprocessing import get_subgraph_living_creature
from settings import dataset_folder, verbose, df_nodes_core_periphery_classification_path, df_core_periphery_generality_living_proportion_path, df_core_periphery_generality_all_proportion_path

mode = 'all'

def main():
    G_dataset = load_dataset(dataset_folder)
    nodes_core_periphery_classification_df = pd.read_pickle(df_nodes_core_periphery_classification_path)
    core_periphery_generality_proportion_df = {'Network': [], 
                                               'Generalist in core proportion': [],
                                               'Vulnerable in core proportion': [],
                                               'Generalist in periphery proportion': [],
                                               'Vulnerable in periphery proportion': []}
    # Setting the output
    if mode == 'living':
        core_periphery_generality_df_path = df_core_periphery_generality_living_proportion_path
    elif mode == 'all':
        core_periphery_generality_df_path = df_core_periphery_generality_all_proportion_path
    else:
        raise ValueError(f'Mode {mode} not implemented')
    if verbose:
        print_time(f"Selected mode: {mode}.")
        if mode == 'living':
            print(f"\tThe proportion of generalist and vulnerable species in the core - periphery structure is computed for the living creature subgraph.")
        elif mode == 'all':
            print(f"\tThe proportion of generalist and vulnerable species in the core - periphery structure is computed for the whole graph.")
        else:
            raise ValueError(f'Mode {mode} not implemented. But how did we get here?')
        print_time(f"Output path: {core_periphery_generality_df_path}.")
        print_time(f"Start")
        start_time = datetime.now().timestamp()
    for i_G, G in enumerate(sorted(G_dataset, key = lambda x: x['name'])):
        if verbose:
            print_progress("Evaluating the proportion of generalist and vulnerable species in the core - periphery structure", i_G, len(G_dataset), n_char = 20)
        # Get nodes
        G = get_subgraph_living_creature(G) if mode == 'living' else G
        L = G.ecount()
        S = G.vcount()
        L_S = L / S
        # Measure generalism and vulnerability for each node
        node_names = G.vs['name']
        node_gen = [v.indegree() / L_S for v in G.vs]
        node_vul = [v.outdegree() / L_S for v in G.vs]
        eco = [v['ECO'] for v in G.vs]
        gen_vul_G_df = pd.DataFrame({'Node name': node_names, 'Gen': node_gen, 'Vul': node_vul, 'ECO': eco}).set_index('Node name')
        # Get core periphery classification for Network G
        c_p_G_df = nodes_core_periphery_classification_df[nodes_core_periphery_classification_df['Network'] == G['name']][['Node name', 'Core periphery']]
        c_p_G_df.set_index('Node name', inplace = True)
        c_p_G_df = c_p_G_df.join(gen_vul_G_df, how = 'outer')
        c_p_G_df['Network'] = G['name']
        # Post processing 1: dropna, reset index and exclude detrital compartments
        c_p_G_df = c_p_G_df.dropna()
        c_p_G_df = c_p_G_df.reset_index()
        c_p_G_df = c_p_G_df[c_p_G_df['ECO'] == 1]
        # Post processing 2: map core periphery classification: Core -> Core, anything else -> Periphery
        c_p_G_df['Core periphery'] = np.where(c_p_G_df['Core periphery'] == 'Core', 'Core', 'Periphery')
        # Post processing 3: classification of generalism and vulnerability
        c_p_G_df['Generalism'] = np.where(c_p_G_df['Gen'] > 1.0, 'General', 'Non-general')
        c_p_G_df['Vulnerability'] = np.where(c_p_G_df['Vul'] > 1.0, 'Vulnerable', 'Non-vulnerable')
        # Post processing 4: measuring proportion of generalist and vulnerable nodes
        N_core = c_p_G_df[c_p_G_df['Core periphery'] == 'Core'].shape[0]
        N_periphery = c_p_G_df[c_p_G_df['Core periphery'] == 'Periphery'].shape[0]
        N_gen_core = c_p_G_df[(c_p_G_df['Core periphery'] == 'Core') & (c_p_G_df['Generalism'] == 'General')].shape[0]
        N_gen_periphery = c_p_G_df[(c_p_G_df['Core periphery'] == 'Periphery') & (c_p_G_df['Generalism'] == 'General')].shape[0]
        N_vul_core = c_p_G_df[(c_p_G_df['Core periphery'] == 'Core') & (c_p_G_df['Vulnerability'] == 'Vulnerable')].shape[0]
        N_vul_periphery = c_p_G_df[(c_p_G_df['Core periphery'] == 'Periphery') & (c_p_G_df['Vulnerability'] == 'Vulnerable')].shape[0]
        
        # Add to list
        gen_core_proportion = N_gen_core / N_core
        gen_periphery_proportion = N_gen_periphery / N_periphery if N_periphery > 0 else 0
        vul_core_proportion = N_vul_core / N_core
        vul_periphery_proportion = N_vul_periphery / N_periphery if N_periphery > 0 else 0
        core_periphery_generality_proportion_df['Network'].append(G['name'])
        core_periphery_generality_proportion_df['Generalist in core proportion'].append(gen_core_proportion)
        core_periphery_generality_proportion_df['Vulnerable in core proportion'].append(vul_core_proportion)
        core_periphery_generality_proportion_df['Generalist in periphery proportion'].append(gen_periphery_proportion)
        core_periphery_generality_proportion_df['Vulnerable in periphery proportion'].append(vul_periphery_proportion)
    if verbose:
        print_progress("Evaluating the proportion of generalist and vulnerable species in the core - periphery structure", len(G_dataset), len(G_dataset), n_char = 20)
        print()
        print_time(f"End")
        print_time(f"Elapsed time: {datetime.now().timestamp() - start_time:.4f} seconds")
    core_periphery_generality_proportion_df = pd.DataFrame(core_periphery_generality_proportion_df)
    core_periphery_generality_proportion_df.to_pickle(core_periphery_generality_df_path)
    if verbose:
        print_time(f"Dataframe successfully saved at: {core_periphery_generality_df_path}.")
if __name__ == "__main__":
    main()