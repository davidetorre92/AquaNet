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
core_periphery_same_habitat_image_out_path = config.get('plots', 'core_periphery_same_habitat_image_out_path')
network_periphery_size_df = pd.read_pickle(network_periphery_size_df_path)

plot_periphery_df = network_periphery_size_df.copy()
N_graph = plot_periphery_df.shape[0]
graph_name_index = range(1,N_graph+1)

#@title Periphery distribution of same habitats foodwebs
import re
def remove_year(name):
    pattern = r'\(\d{4}\)'
    return re.sub(pattern, '', name).strip()

image_out_path = '/content/core_periphery_same_habitat.eps' #@param {type: "string"}
create_folder(image_out_path)

original_names = network_bowtie_df.reset_index(names = 'Graph Name')['Graph Name'].tolist()
strip_names = [remove_year(name) for name in original_names]
index_dict = {}
for index, name in enumerate(strip_names):
    if name in index_dict:
        # Append the index to the existing list for this name
        index_dict[name].append(index)
    else:
        # Create a new list with the current index for this name
        index_dict[name] = [index]

index_dict = {key: item for key, item in index_dict.items() if len(item) > 1}
# Initialize an empty list to store the merged result
indices = []

# Iterate over the dictionary and extend the merged list with each value
for value in index_dict.values():
    indices.extend(value)

season_core_periphery_df = network_bowtie_df.reset_index(names = 'Graph Name')
season_core_periphery_df['Graph Name'] = strip_names
plot_periphery_df = season_core_periphery_df.iloc[indices,:].groupby('Graph Name').mean()
std_periphery_df = season_core_periphery_df.iloc[indices,:].groupby('Graph Name').std()
N_graph = plot_periphery_df.shape[0]
graph_name_index = plot_periphery_df.index.to_list()
core_per_proportions = {col: plot_periphery_df[col].values * 100 for col in ['Core', 'In', 'Out', 'T-In']}

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 10}
matplotlib.rc('font', **font)
fig, ax = plt.subplots(figsize = (10,6))

width_bar_x = 1
bottom = np.zeros(N_graph)
colors = sns.color_palette("viridis", as_cmap=False, n_colors=5)
colors[2] = (0.926579, 0.854645, 0.123353)
# colors[3] = colors[-1]
icolor = 0
for boolean, weight_count in core_per_proportions.items():
    p = ax.bar(graph_name_index, weight_count, width_bar_x, label=boolean, bottom=bottom, color=colors[icolor], edgecolor='white', rasterized=True)
    bottom += weight_count
    icolor += 1

for col in ['Core', 'In', 'Out', 'T-In']:
    # Calculate the cumulative sum for positioning the STD line correctly
    cumulative_sum = np.cumsum([core_per_proportions.get(c, np.zeros(N_graph)) for c in ['Core', 'In', 'Out', 'T-In']], axis=0)
    # Get the current height for each bar to position the STD line
    current_height = cumulative_sum[list(core_per_proportions.keys()).index(col)]
    # Plot the STD line
    ax.errorbar(graph_name_index, current_height, yerr=std_periphery_df[col].values * 100, fmt='k_', ecolor='black', elinewidth=1, capsize=3)

ax.set_title("Network core and periphery variability in food webs in different years")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax.set_xlabel('Graph name')
ax.set_ylabel('Structure component percentage')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.xticks(rotation=45)

fig.tight_layout()
fig.savefig(image_out_path, format='eps')
fig.show()

save_image(fig, core_periphery_same_habitat_image_out_path)