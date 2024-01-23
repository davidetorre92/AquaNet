import numpy as np
import pandas as pd

import os
import argparse
from configparser import ConfigParser
from ..utils.file_handlers import save_image
from ..utils.visualizzations import *

import matplotlib
import matplotlib.ticker as mtick

# ArgParser
parser = argparse.ArgumentParser(description = "Create similarity dataset")
parser.add_argument("--config", "-c", type = str, help = "path/to/config.ini", required=True)
args = parser.parse_args()
config_path = args.config

# Reading settings
config = ConfigParser()
config.read(config_path)

dataset_path = config.get('dataset', 'dataset_path')
network_periphery_size_df_path = config.get('core and periphery', 'network_periphery_size_df_path')
core_periphery_image_out_path = config.get('plots', 'core_periphery_image_out_path')
network_periphery_size_df = pd.read_pickle(network_periphery_size_df_path)

plot_periphery_df = network_periphery_size_df.copy()
print(plot_periphery_df)
N_graph = plot_periphery_df.shape[0]
graph_name_index = range(1,N_graph+1)

# Preprocess for visualization purposes
for column in plot_periphery_df.columns:
    if column != 'graph_name':  # Skip 'Vertices count' column
        plot_periphery_df[column] = plot_periphery_df[column] / plot_periphery_df['Vertices count']

plot_periphery_df.drop(columns = ['Vertices count'], inplace = True)
plot_periphery_df.columns = ['graph_name', 'Core', 'In', 'Out', 'Tubes', 'T-In', 'T-Out', 'Disconnected set']
plot_periphery_df.sort_values(by = 'Core', ascending = False, inplace = True)
print(plot_periphery_df)
core_per_proportions = {col: plot_periphery_df[col].values * 100 for col in ['Core', 'In', 'Out', 'T-In']}

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 10}
matplotlib.rc('font', **font)
fig, ax = plt.subplots(figsize = (10,6))

width_bar_x = 1
bottom = np.zeros(N_graph)
# Use Matplotlib's 'viridis' colormap
cmap = plt.cm.viridis

# Get 5 colors from the 'viridis' colormap
colors = cmap(np.linspace(0, 1, 5))
colors[2] = (0.926579, 0.854645, 0.123353, 1)
# colors[3] = colors[-1]
icolor = 0
for boolean, weight_count in core_per_proportions.items():

    p = ax.bar(graph_name_index, weight_count, width_bar_x, label=boolean, bottom=bottom, color=colors[icolor], edgecolor='none', rasterized=True)
    bottom += weight_count
    icolor += 1

ax.set_title("Network core and periphery proportion among network")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax.set_xlabel('Graph index')
ax.set_ylabel('Structure component percentage')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

fig.tight_layout()
save_image(fig, core_periphery_image_out_path)