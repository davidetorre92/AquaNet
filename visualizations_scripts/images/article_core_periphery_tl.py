import sys  
sys.path.append('/home/davide/AI/Projects/AquaNet') # Initialize Experiment

import os
import numpy as np
import pandas as pd
import json

from utils.data_handling import create_folder, print_time
from settings import verbose, article_images_folder, df_nodes_core_periphery_classification_path
import seaborn as sns
import matplotlib.pyplot as plt
# Deactivate warnings
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.size":30,
    "font.family":"serif",
    "font.weight":"normal",
    "axes.labelsize":20,
    "xtick.labelsize":20,
    "ytick.labelsize":20,
    "legend.fontsize":20
})

df = pd.read_pickle(df_nodes_core_periphery_classification_path)
df['Category'] = df['Core periphery'].apply(lambda x: 'Core' if x == 'Core' else 'Periphery')
print(df.head())
df_all = df.copy()
df_all['Category'] = 'Network'
df = pd.concat([df_all, df])
# Make a boxplot where X is Category. Set the color to red for "Generalist" and blue for "Vulnerable".
fig, ax = plt.subplots(figsize = (8,8))
sns.boxplot(data = df, x='Category', y='Trophic level', 
        showmeans=True, meanprops={'marker':'+' ,'markersize':12, 'markeredgecolor':'black', 'markerfacecolor':'white', 'markeredgewidth': 3}, 
        hue = 'Category', palette=['#00025B', '#FF8200', '#FFC100'], 
        width = 0.25,
        ax = ax)
# Formatting the plot
ax.set_title(f'Trophic Level Composition')
ax.set_ylabel('Trophic Level')
ax.set_xlabel('')

# Display the plot
fig.tight_layout()
# Save the figure
img_path = os.path.join(article_images_folder, 'max_trophic_level_composition.eps')
create_folder(img_path, verbose = verbose)
fig.savefig(img_path)
if verbose:
    print_time(f"Image saved in {img_path}")