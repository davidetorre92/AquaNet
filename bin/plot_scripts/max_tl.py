import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse
from configparser import ConfigParser
from ..utils.visualizzations import boxplot_melt_data

# ArgParser
parser = argparse.ArgumentParser(description = "Create similarity dataset")
parser.add_argument("--config", "-c", type = str, help = "path/to/config.ini", required=True)
args = parser.parse_args()
config_path = args.config

# Reading settings
config = ConfigParser()
config.read(config_path)
max_tl_structure_df_path = config.get('core and periphery', 'max_tl_structure_df_path')
tl_composition_image_out_path = config.get('plots', 'tl_composition_image_out_path')

# Load data
max_tl_df = pd.read_pickle(max_tl_structure_df_path)
max_structure = max_tl_df.reset_index()[['Network', 'Structure', 'Max Trophic Level']]
max_network = max_tl_df.reset_index().set_index('Network')[['Max Trophic Level']].groupby('Network').max().reset_index()
max_network['Structure'] = 'Network'
plot_df = pd.concat([max_network, max_structure]).set_index('Network')
plot_df['Structure'] = plot_df['Structure'].apply(lambda x: 'Core' if x == 'Core' else ('Network' if x == 'Network' else 'Periphery'))
plot_df.dropna(inplace = True)
fig, ax = plt.subplots(figsize=(8,8))
fig, ax = boxplot_melt_data(plot_df, ax = ax, var_name='Structure', value_name = 'Max Trophic Level')
ax.set_ylabel('Max trophic level', fontsize = 24)
ax.tick_params(axis='both', which='major', labelsize=18)
fig.tight_layout()
fig.savefig(tl_composition_image_out_path)
print(f'Image saved in {tl_composition_image_out_path}')