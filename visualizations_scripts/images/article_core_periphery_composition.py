import sys
sys.path.append('/home/davide/AI/Projects/AquaNet')
import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mtick
from utils.data_handling import print_time

from settings import verbose, article_images_folder, df_nodes_core_periphery_classification_path

df_nodes_core_periphery_classification = pd.read_pickle(df_nodes_core_periphery_classification_path)
plot_periphery_df = {'Core': [], 'In': [], 'Out': [], 'Tendrils-In': []}
for network in df_nodes_core_periphery_classification.Network.unique():
    df_network = df_nodes_core_periphery_classification[df_nodes_core_periphery_classification.Network == network]
    vertices = df_network.shape[0]
    core = df_network[df_network['Core periphery'] == 'Core'].shape[0]
    in_periphery = df_network[df_network['Core periphery'] == 'IN set'].shape[0]
    out_periphery = df_network[df_network['Core periphery'] == 'OUT set'].shape[0]
    t_in_periphery = df_network[df_network['Core periphery'] == 'Tendrils IN'].shape[0]
    plot_periphery_df['Core'].append(core / vertices)
    plot_periphery_df['In'].append(in_periphery / vertices)
    plot_periphery_df['Out'].append(out_periphery / vertices)
    plot_periphery_df['Tendrils-In'].append(t_in_periphery / vertices)

N_graph = len(plot_periphery_df['Core'])
network_index = np.arange(N_graph)
plot_periphery_df = pd.DataFrame(plot_periphery_df).sort_values(by = 'Core', ascending = False)
core_per_proportions = {col: plot_periphery_df[col].values * 100 for col in ['Core', 'In', 'Out', 'Tendrils-In']}

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 10}
matplotlib.rc('font', **font)
cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, 5))
colors[2] = colors[-1] # (0.926579, 0.854645, 0.123353, 1)

fig, ax = plt.subplots(figsize = (10,6))

width_bar_x = 1
bottom = np.zeros(N_graph)
icolor = 0
for boolean, weight_count in core_per_proportions.items():
    p = ax.bar(network_index, weight_count, width_bar_x, label=boolean, bottom=bottom, color=colors[icolor], edgecolor='none', rasterized=True)
    bottom += weight_count
    icolor += 1

ax.set_title("Network core and periphery proportion among network")
ax.legend(loc="lower left", bbox_to_anchor=(0.05, 0.05))
ax.set_xlabel('Graph index')
ax.set_ylabel('Structure component percentage')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

fig.tight_layout()
core_periphery_image_out_path = os.path.join(article_images_folder, 'core_periphery_results.eps')
df_core_periphery_proportion = pd.DataFrame(core_per_proportions)
print(df_core_periphery_proportion.describe())
print("Percentage of networks with less than 50\% of nodes in the core:")
print(df_core_periphery_proportion[df_core_periphery_proportion['Core'] < 50].shape[0] / (df_core_periphery_proportion.shape[0] + 0.) * 100)
print("For a total of", df_core_periphery_proportion[df_core_periphery_proportion['Core'] < 50].shape[0], "networks")
fig.savefig(core_periphery_image_out_path)
if verbose:
    print_time(f"Image saved in {core_periphery_image_out_path}")