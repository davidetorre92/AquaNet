import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

import igraph as ig
import pdb

def load_dataset(dataset_folder, name = None, verbose = False):
    G_dataset = []
    for file in os.listdir(dataset_folder):
        if file.endswith(".graphml"):
            if verbose: print(f"Reading {file}")
            G = ig.Graph.Read_GraphML(os.path.join(dataset_folder, file))
            if verbose:
                print(G.summary())
            if name is not None:
                if name in file:
                    return G
            else:
                G_dataset.append(G)
    if name is not None:
        raise ValueError(f"Graph {name} not found.")
    else:
        return G_dataset

def print_time(string, fp = None):
    string = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {string}"
    if fp is None:
        print(string)
    else:
        fp.write(string)
    return 1

def print_progress(title_string, i, max, status_str = "", n_char = 100, null_str = '-', adv_str = '#'):
    n_adv = int((i / max) * n_char)
    n_null = n_char - n_adv
    string = f"{title_string}"
    string += f" |{adv_str * n_adv}{null_str * n_null}| {i}/{max}"
    print(string, end = '\r')
    return

def create_folder(path, is_file = True, verbose = False):
    # If path is a file with extension, extract folder from filename
    if is_file: folder = os.path.dirname(path)
    else: folder = path
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        print_time(f"Created folder {folder}")
        return 1
    else:
        if verbose: print_time(f"Folder {folder} already exists")
        return 0
    
def get_basefilename(G_name):
    lower_text = os.path.splitext(G_name)[0].lower()
    splitted_text = [word for word in re.sub('[^\w]', '_', lower_text).split('_') if word != '']
    basefilename = '_'.join(splitted_text)
    return basefilename