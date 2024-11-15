import sys
sys.path.append('/home/davide/AI/Projects/AquaNet') # Initialize Experiment
import numpy as np
import pandas as pd
from utils.data_handling import create_folder

from settings import top_3_critical_nodes_path, df_critical_nodes_path, df_robustness_path, verbose

N = 3
print_rho_flag = False

create_folder(top_3_critical_nodes_path, verbose = verbose)
critical_nodes_df = pd.read_pickle(df_critical_nodes_path)
robustness_df = pd.read_pickle(df_robustness_path)
merged_df = pd.merge(critical_nodes_df, robustness_df, on='Network')
merged_df.sort_values(by=['Network', 'seq_index'], ascending=True, inplace=True)
# Group names by 'Network' and sort by 'rho_G'
grouped = merged_df.groupby(['Network', 'rho_G'])['name'].apply(lambda x: '; '.join(x[:N])).reset_index()

# Sort by rho_G in descending order
grouped = grouped.sort_values(by='rho_G', ascending=False)
grouped['Rank'] = range(1, grouped.shape[0] + 1)
# Order and rename columns
if print_rho_flag:
    grouped = grouped[['Rank', 'Network', 'name', 'rho_G']]
    grouped.columns = ['Rank', 'Food web', f"[{','.join([f'$s_{i+1}^G$' for i in range(N)])}]", '$\rho_G$']
else:
    grouped = grouped[['Rank', 'Network', 'name']]
    grouped.columns = ['Rank', 'Food web', f"[{','.join([f'$s_{i+1}^G$' for i in range(N)])}]"]
# Generate LaTeX table
grouped.to_latex(top_3_critical_nodes_path, index = False)
print(f"LaTeX code successfully saved to {top_3_critical_nodes_path}")