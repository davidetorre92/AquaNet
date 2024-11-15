import sys  
sys.path.append('/home/davide/AI/Projects/AquaNet') # Initialize Experiment

import os 
import numpy as np
import pandas as pd
from utils.data_handling import create_folder, print_time
from settings import verbose, motif_Z_scores_df_path, article_images_folder
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import seaborn as sns

fonts = {'family': 'serif',
    'weight': 'normal'}
matplotlib.rc('font', **fonts)
# Define the custom colormap
colors = [
    (1.0, 0.5098039215686274, 0.0),  # Orange (negative values)
    (0.4196078431372549, 0.6039215686274509, 0.7686274509803922),  # Grey (zeros)
    (0.0, 0.00784313725490196, 0.3568627450980392), # Dark Blue (positive values)
]
custom_cmap = LinearSegmentedColormap.from_list("BlueOrange", colors, N=256)
y_lab_str = 'Profile of the Z-score values'
x_lab_str = 'Motif name'

create_folder(article_images_folder, is_file = False, verbose = verbose)
df = pd.read_pickle(motif_Z_scores_df_path)

df.drop('Network', axis=1, inplace=True)
columns = df.columns
rows = []
for i, row in df.iterrows():
    sum_sq = np.sqrt((row * row).sum())
    rows.append(row / sum_sq)

df_profile_z_score = pd.DataFrame(rows, columns = columns)

# Box plot
fig, ax = plt.subplots(figsize = (16,9))

sns.boxplot(df_profile_z_score, ax = ax, color = '#6B9AC4', flierprops={"marker": "d", "color": "black"})
ax.set_ylim((-1,1))
for x in range(0, len(columns), 2):
    patch = Rectangle((x - 0.5,-1), 1, 2, color = (0.7,0.7,0.7,0.2))
    ax.add_patch(patch)
ax.grid(which = 'major', axis = 'y')
ax.axhline(0, color = 'black')
ax.set_title('Profile of the Z-score', fontsize = 20)
ax.set_xlabel(x_lab_str, fontsize = 15)
ax.set_ylabel(y_lab_str, fontsize = 15)

img_path = os.path.join(article_images_folder, 'profile_z_score.eps')
fig.savefig(img_path)
if verbose:
    print_time(f"Image saved in {img_path}")

df_profile_z_score['Family'] = df['S2'].apply(lambda x: 'S2 over-reppresented' if x > 0 else 'S2 under-reppresented')
df_profile_z_score = df_profile_z_score.melt(id_vars='Family', value_name=y_lab_str, var_name=x_lab_str)
# Box plot
fig, ax = plt.subplots(figsize = (16,9))

colors = ["#00025B", "#FF8200"]
sns.set_palette(sns.color_palette(colors))
sns.boxplot(data = df_profile_z_score, x = x_lab_str, y = y_lab_str, hue = 'Family', ax = ax, flierprops={"marker": "d", "color": "black"})
ax.set_ylim((-1,1))
for x in range(0, len(columns), 2):
    patch = Rectangle((x - 0.5,-1), 1, 2, color = (0.7,0.7,0.7,0.2))
    ax.add_patch(patch)
ax.grid(which = 'major', axis = 'y')
ax.axhline(0, color = 'black')
ax.set_title('Profile of the Z-score', fontsize = 20)
ax.set_ylabel('Profile of the Z-score values', fontsize = 15)
ax.set_xlabel('Motif name', fontsize = 15)

img_path = os.path.join(article_images_folder, 's2_profile.eps')
fig.savefig(img_path)
if verbose:
    print_time(f"Image saved in {img_path}")
