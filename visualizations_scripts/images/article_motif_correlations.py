import sys  
sys.path.append('/home/davide/AI/Projects/AquaNet') # Initialize Experiment

import os
import numpy as np
import pandas as pd
import json

from utils.measurements import get_original_motif_count, get_random_ensamble_motif_count, calculate_z_score
from utils.data_handling import create_folder, print_time
from settings import article_images_folder, motif_Z_scores_df_path, verbose
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "font.size": 20,
    "font.family": "serif",
    "font.weight": "light",
    "axes.labelsize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 20,
    "figure.titlesize": 25
})
# Define the custom colormap
colors = [
    (1.0, 0.5098039215686274, 0.0),  # Orange (negative values)
    (0.4196078431372549, 0.6039215686274509, 0.7686274509803922),  # Grey (zeros)
    (0.0, 0.00784313725490196, 0.3568627450980392), # Dark Blue (positive values)
]
custom_cmap = LinearSegmentedColormap.from_list("BlueOrange", colors, N=256)

create_folder(article_images_folder, is_file = False, verbose = verbose)
df = pd.read_pickle(motif_Z_scores_df_path)
df = df[['S1', 'S2', 'S4', 'S5']]
rename_dictionary = {'S1': r'$S_1$' + '\nTri-trophic chain', 
                     'S2': r'$S_2$' + '\nIntraguild predation', 
                     'S4': r'$S_4$' + '\nApparent competition', 
                     'S5': r'$S_5$' + '\nExploitative competition'}
df = df.rename(columns = rename_dictionary)
fig, ax = plt.subplots(figsize = (9,9), tight_layout = True)
correlation_df = df.corr(numeric_only = True)
pos = ax.imshow(correlation_df, cmap=custom_cmap, interpolation='none', vmin=-1, vmax=1)
# Add xlabel and ylabel
ax.set_xticks(range(len(correlation_df.index)), correlation_df.index, rotation = 45, ha = 'right')
ax.set_yticks(range(len(correlation_df.columns)), correlation_df.columns)
ax.set_title("Motif representation correlation matrix", fontsize = 20)
# Set colorbar
fig.colorbar(pos, ax = ax, fraction = 0.04, pad = 0.04)

# Annotate
for i in range(len(correlation_df.index)):
    for j in range(len(correlation_df.columns)):
        value = round(correlation_df.iloc[i, j], 2)
        color = 'black' if value < 0.25 and value > -0.25 else 'white' 
        text = ax.text(j, i, value, ha = 'center', va = 'center', color = color, fontsize = 20)
path = os.path.join(article_images_folder, 'correlation_heatmap.eps')
fig.savefig(path, dpi = 300)
if verbose: print_time(f"Image saved in {path}")