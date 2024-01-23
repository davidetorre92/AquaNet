import os
import numpy as np
import igraph as ig
import warnings
import sys

def get_dataset(dataset_path, verbose = False, living_other = True):
  warnings.filterwarnings('ignore', message="Could not add vertex ids, there is already an 'id' vertex attribute", category=RuntimeWarning)

  G_dataset = []
  for file_name in sorted(os.listdir(dataset_path)):
    if file_name.endswith(".graphml"):
      file_path = os.path.join(dataset_path, file_name)
      if verbose: print(f"Reading {file_path}")
      G = ig.Graph.Read_GraphML(file_path)
      if verbose: print(f"{G['name']}", end=' ')
      if living_other:
        keep = list(np.argwhere(np.array(G.vs()['ECO']) == 1.0).ravel()) + list(np.argwhere(np.array(G.vs()['ECO']) == 2.0).ravel())
        G = G.subgraph(keep)
        degree_keep = G.degree()
        keep = list(np.argwhere(np.array(degree_keep) > 0).ravel())
        G = G.subgraph(keep)
      G_dataset.append(G)
      if verbose: print(f"{file_name} read correctly ✅")
  return G_dataset

def get_dataset_living(dataset_path, verbose = False):
  G_dataset = get_dataset(dataset_path)
  G_living_compartment_dataset = []
  for G in G_dataset:
    v_sub = [v for v in G.vs() if v['ECO'] == 1]
    G_living_compartment_dataset.append(G.subgraph(v_sub))
  return G_living_compartment_dataset

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

def save_df_to_pickle(df, path):
  create_folder(path, verbose = True)
  try:
      df.to_pickle(path)
      print(f"Table saved in: {path}.")
  except Exception as e:
      # Handle any exceptions (e.g., permission errors)
      print(f"An error occurred while saving the file {path}: {e}")

def progression_bar(current_position, max_position):

  # Progress bar settings
  bar_length = 50
  progress = current_position / max_position
  block = int(round(bar_length * progress))

  # Create the progress bar string
  bar = '[' + '=' * block + '-' * (bar_length - block) + ']'

  # Print the progress message with the progress bar
  sys.stdout.write(f'\rFood Web Graph Processed: {current_position} / {max_position} {bar}')
  sys.stdout.flush()
