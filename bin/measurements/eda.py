import numpy as np
import pandas as pd

import os
import argparse
from configparser import ConfigParser
from ..utils.file_handlers import get_dataset, save_df_to_pickle

# ArgParser
parser = argparse.ArgumentParser(description = "Create similarity dataset")
parser.add_argument("--config", "-c", type = str, help = "path/to/config.ini", required=True)
args = parser.parse_args()
config_path = args.config

# Reading settings
config = ConfigParser()
config.read(config_path)

dataset_path = config.get('dataset', 'dataset_path')
eda_df_path = config.get('eda', 'eda_path')

G_dataset = get_dataset(dataset_path, verbose = False)

rows = []
for G in G_dataset:
  G_liv = G.subgraph([v for v in G.vs() if v['ECO'] == 1])
  det = len([v for v in G.vs() if v['ECO'] == 2])
  basal = len([v for v in G_liv.vs() if v.indegree() == 0])
  S = G.vcount()
  L = G.ecount()

  rows.append((G['name'], S, L, S / (L*L), basal / S, det / S, G.transitivity_avglocal_undirected()))

eda_df = pd.DataFrame(rows, columns = ['Graph name', 'S', 'L', 'C', 'B/N', 'det/N', 'clustering'])

save_df_to_pickle(eda_df, eda_df_path)