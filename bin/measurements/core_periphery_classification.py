import numpy as np
import pandas as pd

import os
import argparse
from configparser import ConfigParser
from ..utils.file_handlers import get_dataset, save_df_to_pickle
from ..utils.core_periphery import get_core_periphery_structure
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
nodes_classification_df_path = config.get('core and periphery', 'nodes_classification_df_path')

G_dataset = get_dataset(dataset_path, verbose = False)
# Nodes classification
rows_nodes = []

rows_size = []
for G in G_dataset:
    graph_name = G['name']
    core_periphery_dict = get_core_periphery_structure(G)
    for periphery_structure_name, node_set in core_periphery_dict.items():
        for node_id in node_set:
            node_name = G.vs[node_id]['name']
            eco = G.vs[node_id]['ECO']
            row_node = (graph_name, node_id, node_name, eco, periphery_structure_name)
            rows_nodes.append(row_node)
    row_size = (graph_name,
                len(core_periphery_dict['Core']),
                len(core_periphery_dict['IN set']),
                len(core_periphery_dict['OUT set']),
                len(core_periphery_dict['Tubes']),
                len(core_periphery_dict['Tendrils IN']),
                len(core_periphery_dict['Tendrils OUT']),
                len(core_periphery_dict['Disconnected set']),
                G.vcount()
    )
    rows_size.append(row_size)

columns_nodes = ['graph_name', 'node_id', 'node_name', 'ECO', 'periphery_structure_name']
columns_size = ['graph_name', 'Core', 'IN set', 'OUT set', 'Tubes', 'Tendrils IN', 'Tendrils OUT', 'Disconnected set', 'Vertices count']
# Save
nodes_classification_df = pd.DataFrame(rows_nodes, columns = columns_nodes)
network_periphery_size_df = pd.DataFrame(rows_size, columns = columns_size)

save_df_to_pickle(nodes_classification_df, nodes_classification_df_path)
save_df_to_pickle(network_periphery_size_df, network_periphery_size_df_path)



