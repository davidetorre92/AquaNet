import os
import numpy as np
import igraph as ig
from .connectiveness import f_G

def get_dataset(dataset_path, verbose = True, living_other = True):
  G_dataset = []
  for file_name in sorted(os.listdir(dataset_path)):
    if file_name.endswith(".graphml"):
      file_path = os.path.join(dataset_path, file_name)
      G = ig.Graph.Read_GraphML(file_path)
      if verbose: print(f"{G['name']}", end=' ')
      if f_G(G) > 0 and len([v for v in G.vs if v['ECO'] == 2]) != 0:
        if living_other:
          keep = list(np.argwhere(np.array(G.vs()['ECO']) == 1.0).ravel()) + list(np.argwhere(np.array(G.vs()['ECO']) == 2.0).ravel())
          G = G.subgraph(keep)
          degree_keep = G.degree()
          keep = list(np.argwhere(np.array(degree_keep) > 0).ravel())
          G = G.subgraph(keep)
        G_dataset.append(G)
        if verbose: print(f"{file_name} read correctly ✔️")
      else:
          if verbose: print(f"Skipping {G['name']} (either due to 0 connectivity value or few nodes) ❌")
  return G_dataset

def index_from_dataset(G_dataset, name):
  for i, G in enumerate(G_dataset):
    if G['name'] == name:
      return i
  print(f"{name} not found in dataset...")
  return None

def create_folder(path, verbose=True):
    """
    Creates a folder and any intermediate directories for the given path if they don't already exist.
    Parameters:
        path (str): The file path where the folder should be created.
        verbose (bool): If True, the function will print messages about its operations.
    """
    folder = os.path.dirname(os.path.abspath(path))
    try:
        # Create the directory, also create intermediate directories if necessary
        os.makedirs(folder, exist_ok=True)
        if verbose:
            print(f"Folder {folder} created or already exists.")
    except Exception as e:
        # Handle any exceptions (e.g., permission errors)
        if verbose:
            print(f"An error occurred while creating the folder {folder}: {e}")
