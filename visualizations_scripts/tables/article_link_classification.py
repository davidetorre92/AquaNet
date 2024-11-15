import sys  
sys.path.append('/home/davide/AI/Projects/AquaNet') # Initialize Experiment

import os
import numpy as np
import pandas as pd
from utils.data_handling import print_time, create_folder
from settings import verbose, df_links_core_periphery_classification_path, article_table_folder
import matplotlib.pyplot as plt

def main():
    # Read DataFrames
    df = pd.read_pickle(df_links_core_periphery_classification_path)
    df['Structure to structure'] = list(zip(df['Source core periphery'], df['Target core periphery']))
    df['Interaction type'] = ['LS -> LS' if x == (1,1)
                            else ('LS -> D' if x == (1,2)
                            else ('D -> LS' if x == (2,1)
                            else 'D -> D')) for x in zip(df['Source ECO'], df['Target ECO'])]
    df_tot = df.pivot_table(index='Interaction type', columns='Structure to structure', aggfunc='size', fill_value=0)
    df_tot['Total'] = df_tot.sum(axis=1)
    column_sums = df_tot.sum(axis=0).to_frame().T
    column_sums.index = ['Total']
    df_tot = pd.concat([df_tot, column_sums])
    # Change columns order
    df_tot = df_tot[[('IN set', 'IN set'), ('IN set', 'Core'), ('IN set', 'Tendrils IN'), 
                     ('Core', 'Core'), ('Core', 'OUT set'), ('OUT set', 'OUT set'), ('IN set', 'OUT set'), 'Total']]
    # Change index order
    df_tot = df_tot.reindex(index = ['LS -> LS', 'LS -> D', 'D -> LS', 'D -> D', 'Total'])
    if verbose: print(df_tot)
    create_folder(article_table_folder, is_file = False, verbose = verbose)
    table_path = os.path.join(article_table_folder, 'link_classification.tex')
    df_tot.to_latex(table_path, escape=False)
    if verbose: 
        print_time(f"Table saved in {table_path}")
if __name__ == '__main__':
    main()