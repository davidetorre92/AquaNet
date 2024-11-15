import sys  
sys.path.append('/home/davide/AI/Projects/AquaNet') # Initialize Experiment

import pandas as pd
from utils.data_handling import load_dataset, print_time, create_folder
from utils.preprocessing import nodes_difference_analysis, edges_difference_analysis, core_periphery_difference_analysis
from settings import verbose, dataset_folder, sm_table_folder, df_nodes_core_periphery_classification_path

def main():
    # Read DataFrames
    df = pd.read_pickle(df_nodes_core_periphery_classification_path)
    G_dataset = load_dataset(dataset_folder)
    # Collect food webs at different seasons
    seasonal_datasets = []
    for G in G_dataset:
        name_G = G['name']
        if 'season' in name_G.lower():
            seasonal_datasets.append(name_G)
    seasonal_sets = []
    for i_name_1 in range(len(seasonal_datasets)):
        name_1 = seasonal_datasets[i_name_1]
        for i_name_2 in range(i_name_1 + 1, len(seasonal_datasets)):
            name_2 = seasonal_datasets[i_name_2]
            if name_1.split(' ')[0] == name_2.split(' ')[0]:
                seasonal_sets.append((name_1, name_2))
    if verbose:
        seasonal_food_webs = len(seasonal_sets) * 2
        print_time(f"Found {seasonal_food_webs} seasonal food webs")
        print_time("Start")

    output_lines = []
    output_lines += ["\\begin{itemize}\n"]

    for seasonal_set in seasonal_sets:
        # Load graphs
        name_1 = seasonal_set[0]
        name_2 = seasonal_set[1]
        G_1 = load_dataset(dataset_folder, name = name_1)
        G_2 = load_dataset(dataset_folder, name = name_2)
        output_lines += [f"\\item {G_1['name']} vs. {G_2['name']}:\n"]
        output_lines += ["\t\\begin{itemize}\n"]
        result_line = nodes_difference_analysis(G_1, G_2)
        output_lines += result_line
        result_line = edges_difference_analysis(G_1, G_2)
        output_lines += result_line
        result_line = core_periphery_difference_analysis(G_1, G_2, df)
        output_lines += result_line
        output_lines += ["\t\\end{itemize}\n"]
    output_lines += ["\\end{itemize}\n"]           
    create_folder(sm_table_folder, is_file = False, verbose = verbose)
    seasonal_food_webs_report_path = f"{sm_table_folder}/seasonal_food_webs_report.tex"         
    with open(seasonal_food_webs_report_path, 'w') as f:
        f.writelines(output_lines)
    if verbose:
        print_time(f"Done!")
        print_time(f"File saved in {seasonal_food_webs_report_path}")
        
if __name__ == '__main__':
    main()