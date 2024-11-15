import sys  
sys.path.append('/home/davide/AI/Projects/AquaNet') # Initialize Experiment

import os
import numpy as np
import pandas as pd
from utils.data_handling import create_folder, print_time
from settings import verbose, df_biome_path, df_motif_Z_scores_path, article_images_folder
import igraph as ig
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    df_biome = pd.read_csv(df_biome_path)
    df_motifs = pd.read_pickle(df_motif_Z_scores_path)
    rows = []
    for network in df_motifs['Network'].unique():
        motifs = df_motifs[df_motifs['Network'] == network]
        z_s1_score = motifs['S1'].values[0]
        z_s2_score = motifs['S2'].values[0]
        z_s4_score = motifs['S4'].values[0]
        z_s5_score = motifs['S5'].values[0]
        biome = df_biome[df_biome['Network'] == network]['Biome']
        if biome.empty:
            if verbose: print(f"Biome not found for {network}")
            continue
        else:
            biome = biome.values[0]
        row = [network, biome, z_s1_score, z_s2_score, z_s4_score, z_s5_score]
        rows.append(row)
    df = pd.DataFrame(rows, columns=['Network', 'Biome', 'S1', 'S2', 'S4', 'S5'])
    df = df.dropna()
    df_s2_representation = df.copy().drop(columns=['Network', 'S1', 'S4', 'S5'])
    df_s2_representation['S2 over represented'] = df['S2'].apply(lambda x: 1 if x > 0 else 0)
    df_s2_representation['S2 under represented'] = df['S2'].apply(lambda x: 1 if x < 0 else 0)
    df_s2_representation['S2 null'] = df['S2'].apply(lambda x: 1 if x == 0 else 0)
    df_s2_representation.drop(columns = ['S2'], inplace = True)
    df_s2_representation_all_mean = df_s2_representation.drop(columns=['Biome']).mean().to_frame().T
    df_s2_representation_all_mean.index = ['All']
    df_s2_representation_biome_mean = df_s2_representation.groupby('Biome').mean()
    df_plot = pd.concat([df_s2_representation_biome_mean, df_s2_representation_all_mean])
    fig, axs = plt.subplots(1,2, figsize=(10, 5))
    sns.heatmap(df_plot, annot = True, ax = axs[0], fmt = '.2%', cmap = 'Blues')
    df_plot_reps = pd.melt(df, id_vars=['Network', 'Biome'], var_name = 'Motif name', value_name = 'Z score')
    axs[0].axhline(0, color = 'black')
    axs[1].axhline(0, color = 'black')
    axs[1].add_patch(plt.Rectangle((-0.25, -30), 0.5, 60, fc="grey", alpha = 0.2))
    axs[1].add_patch(plt.Rectangle((0.75, -30), 0.5, 60, fc="grey", alpha = 0.2))
    axs[1].add_patch(plt.Rectangle((1.75, -30), 0.5, 60, fc="grey", alpha = 0.2))
    axs[1].add_patch(plt.Rectangle((2.75, -30), 0.5, 60, fc="grey", alpha = 0.2))
    sns.lineplot(df_plot_reps, x = 'Motif name', y = 'Z score', 
                 ax = axs[1], hue = 'Biome', 
                 palette = 'tab10', 
                 units = 'Network', estimator = None, linestyle = '--', marker = 'o',
                 hue_order = df.Biome.unique())
    axs[1].set_xlim([-0.2, 3.2])
    axs[1].set_ylim([-25, 25])
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    image_out_path = os.path.join(article_images_folder, 'biome_representation.pdf')
    create_folder(image_out_path, verbose = verbose)
    fig.savefig(image_out_path)
    if verbose:
        print_time(f"Image saved in {image_out_path}")
if __name__ == "__main__":
    main()