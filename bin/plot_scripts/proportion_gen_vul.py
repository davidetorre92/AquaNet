import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse
from configparser import ConfigParser
from ..utils.visualizzations import boxplot_percentages_and_print_data
# ArgParser
parser = argparse.ArgumentParser(description = "Create similarity dataset")
parser.add_argument("--config", "-c", type = str, help = "path/to/config.ini", required=True)
args = parser.parse_args()
config_path = args.config

# Reading settings
config = ConfigParser()
config.read(config_path)
gen_vul_composition_df_path = config.get('core and periphery', 'gen_vul_composition_df_path')
core_composition_image_out_path = config.get('plots', 'core_composition_image_out_path')
periphery_composition_image_out_path = config.get('plots', 'periphery_composition_image_out_path')

# Load data
gen_vul_composition_df = pd.read_pickle(gen_vul_composition_df_path)

# Get core data
print('CORE DATA')
core_df = gen_vul_composition_df[['% gen core', '% vul core']]
core_df.columns = ['Generalist', 'Vulnerable']
fig, ax = plt.subplots(figsize=(8,8))
fig, ax = boxplot_percentages_and_print_data(core_df, ax = ax)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('Core', fontsize=28)
ax.tick_params(axis='both', which='major', labelsize=18)
fig.savefig(core_composition_image_out_path)
print(f'Image saved in {core_composition_image_out_path}')

# Get periphery data
print('PERIPHERY DATA')
periphery_df = gen_vul_composition_df[['% gen periphery', '% vul periphery']]
periphery_df.columns = ['Generalist', 'Vulnerable']
fig, ax = plt.subplots(figsize=(8,8))
fig, ax = boxplot_percentages_and_print_data(periphery_df, ax = ax)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('Periphery', fontsize=28)
ax.tick_params(axis='both', which='major', labelsize=18)
fig.tight_layout()
fig.savefig(periphery_composition_image_out_path)
print(f'Image saved in {periphery_composition_image_out_path}')