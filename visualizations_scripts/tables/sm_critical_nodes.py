import sys
sys.path.append('/home/davide/AI/Projects/AquaNet') # Initialize Experiment
import numpy as np
import pandas as pd

from utils.data_handling import create_folder
from settings import sm_table_folder, df_critical_nodes_path, df_robustness_path

critical_nodes_df = pd.read_pickle(df_critical_nodes_path)
robustness_df = pd.read_pickle(df_robustness_path)
merged_df = pd.merge(critical_nodes_df, robustness_df, on='Network')
merged_df.sort_values(by=['Network', 'seq_index'], ascending=True, inplace=True)
# Drop Nan: experiments not yet available.
merged_df = merged_df.dropna()
# Group names by 'Network' and sort by 'rho_G'
grouped = merged_df.groupby(['Network', 'rho_G'])['name'].apply(lambda x: '; '.join(x)).reset_index()

# Sort by rho_G in descending order
grouped = grouped.sort_values(by='rho_G', ascending=False)

# Generate LaTeX code
latex_entries = []
for _, row in grouped.iterrows():
    network = row['Network']
    rho = row['rho_G']
    species_list = row['name']
    # Substitute each & with \& and each _ with \_
    species_list = species_list.replace('&', '\\&').replace('_', '\\_')
    latex_entries.append(f"\\hline\n"
                         f"\\multicolumn{{1}}{{|c|}}{{\\textbf{{{network}}}, $\\rho = {rho:.6f}$}} \\\\\n"
                         f"\\hline\n"
                         f"{species_list}\\\\\n"
                         f"\\fullhline")

# Combine LaTeX entries into one string
latex_code = "\n".join(latex_entries)

# Output the LaTeX code
create_folder(sm_table_folder, is_file = False, verbose = True)
sm_critical_nodes_extended_table_3_path = f'{sm_table_folder}/sm_table_3_critical_nodes.tex'
with open(sm_critical_nodes_extended_table_3_path, "w") as file:
    file.write(latex_code)

print(f"LaTeX code successfully saved to {sm_critical_nodes_extended_table_3_path}")