import numpy as np

import pandas as pd
import igraph as ig
from random import sample
from collections import OrderedDict
from itertools import combinations

def is_valid_edge_swap(single_edge_list, double_edge_list, edge_1, edge_2):
  A, B = edge_1
  C, D = edge_2

  # self loops?
  if A == B or C == D:
    return False

  # will I create a self loop?
  if A == D or B == C:
    return False
  # are the new edges in the list?
  if ([A,D] in single_edge_list) or ([D,A] in single_edge_list) or ([A,D] in double_edge_list) or ([D,A] in double_edge_list):
    return False

  # are the new edges in the list?
  if ([C,B] in single_edge_list) or ([B,C] in single_edge_list) or ([C,B] in double_edge_list) or ([B,C] in double_edge_list):
    return False

  return True

def get_edge_lists(G):
  edge_list = []
  for e in G.es():
    edge_list.append([e.source, e.target])

  single_edge_list = []
  double_edge_list = []
  for e in G.es():
    source = e.source
    target = e.target
    if source == target: continue # skip self-loops
    if [target, source] in edge_list:
      if [target, source] not in double_edge_list: double_edge_list.append([source, target])
    else:
      single_edge_list.append([source, target])

  return single_edge_list, double_edge_list

def swap_G_v2(G):
  single_edge_list, double_edge_list = get_edge_lists(G)
  n_single_edge = len(single_edge_list)
  n_double_edge = len(double_edge_list)

  if n_single_edge >= 2:
    for n in range(100*n_single_edge):
      iA, iB = sample(range(n_single_edge), 2)
      if is_valid_edge_swap(single_edge_list, double_edge_list, single_edge_list[iA], single_edge_list[iB]) is False: continue
      A, B = single_edge_list[iA]
      C, D = single_edge_list[iB]
      #swap
      single_edge_list[iA][1] = D
      single_edge_list[iB][1] = B

  if n_double_edge >= 2:
    for n in range(100*n_double_edge):
      iA, iB = sample(range(n_double_edge), 2)
      if is_valid_edge_swap(single_edge_list, double_edge_list, double_edge_list[iA], double_edge_list[iB]) is False: continue
      A, B = double_edge_list[iA]
      C, D = double_edge_list[iB]
      #swap
      double_edge_list[iA][1] = D
      double_edge_list[iB][1] = B

  edge_list_swap = [[edge[0], edge[1]] for edge in single_edge_list] + [[edge[0], edge[1]] for edge in double_edge_list] + [[edge[1], edge[0]] for edge in double_edge_list]

  S = G.vcount()
  G_swap = ig.Graph(n=S, edges=edge_list_swap, directed=True)
  return G_swap

def motif_contained_in_G_fast(G):
  igraph_to_nx_translate = {'021C': 'S1',
                            '030T': 'S2',
                            '030C': 'S3',
                            '021U': 'S4',
                            '021D': 'S5',
                            '120U': 'D1',
                            '120D': 'D2',
                            '111U': 'D3',
                            '111D': 'D4',
                            '120C': 'D5',
                            '300': 'D6',
                            '210': 'D7',
                            '201': 'D8'}

  motif_count = {}
  triads = G.triad_census()
  for igraph_motif in igraph_to_nx_translate:
    nx_motif_label = igraph_to_nx_translate[igraph_motif]
    motif_count[nx_motif_label] = triads[igraph_motif]

  return motif_count

def load_motifs():

  H = OrderedDict({
        "S1": ig.Graph(n = 3, edges = [[0,1], [1,2]], directed = True),                                     # Foodchain
        "S2": ig.Graph(n = 3, edges = [[0,1], [2,0], [2,1]], directed = True),                              # Omnivory
        "S3": ig.Graph(n = 3, edges = [[0,1], [1,2], [2,0]], directed = True),                              # Autocatalisys
        "S4": ig.Graph(n = 3, edges = [[0,1], [2,1]], directed = True),                                     # Direct Competition
        "S5": ig.Graph(n = 3, edges = [[1,0], [1,2]], directed = True),                                     # Apparent Competition
        "D1": ig.Graph(n = 3, edges = [[0,1], [1,0], [0,2], [1,2]], directed = True),
        "D2": ig.Graph(n = 3, edges = [[0,1], [1,0], [2,1], [2,0]], directed = True),
        "D3": ig.Graph(n = 3, edges = [[0,1], [1,0], [1,2]], directed = True),
        "D4": ig.Graph(n = 3, edges = [[0,1], [1,0], [2,0]], directed = True),
        "D5": ig.Graph(n = 3, edges = [[0,1], [1,0], [1,2], [2,0]], directed = True),
        "D6": ig.Graph(n = 3, edges = [[0,1], [1,0], [1,2], [2,1], [0,2], [2,0]], directed = True),
        "D7": ig.Graph(n = 3, edges = [[0,1], [1,0], [1,2], [2,1], [0,2]], directed = True),
        "D8": ig.Graph(n = 3, edges = [[0,1], [1,0], [1,2], [2,1]], directed = True)
               })
  for motif in list(H.keys()):
    H[motif].vs()['name'] = [1,2,3]
    H[motif].vs()['ECO'] = [4,4,4]

  return H

def generate_tuples(indices, tuple_length):
    # Use combinations to generate tuples of the specified length
    result = list(combinations(indices, tuple_length))
    return result

def get_node_roles_in_triads(G):
  graph_name = G['name']
  taxa_indices = [v.index for v in G.vs]
  triads = generate_tuples(taxa_indices, 3)
  motifs_dataset = load_motifs()

  # Deleting loops
  edges_no_loops = [(e.source, e.target) for e in G.es if e.source != e.target]
  G_exp = ig.Graph(n = G.vcount(), edges = edges_no_loops, directed=True)
  G_exp.vs['name'] = G.vs['name']
  G_exp.vs['ECO'] = G.vs['ECO']

  rows = []
  for triad in triads:
    G_sub = G_exp.subgraph(triad)
    for motif_name, h in motifs_dataset.items():
      if h.isomorphic(G_sub):
        for sub_id, node_id in enumerate(triad):
          if motif_name == 'S1':
            if G_sub.vs[sub_id].outdegree() == 1 and G_sub.vs[sub_id].indegree() == 1: role = 'middle'
            elif G_sub.vs[sub_id].outdegree() == 1: role = 'bottom'
            elif G_sub.vs[sub_id].indegree() == 1: role = 'top'
            else: print('Are you ok,', motif_name, '?')
          elif motif_name == 'S2':
            if G_sub.vs[sub_id].outdegree() == 2: role = 'bottom'
            elif G_sub.vs[sub_id].outdegree() == 1 and G_sub.vs[sub_id].indegree() == 1: role = 'middle'
            elif G_sub.vs[sub_id].outdegree() == 0: role = 'top'
            else: print('Are you ok,', motif_name, '?')
          elif motif_name == 'S4':
            if G_sub.vs[sub_id].indegree() == 2: role = 'top'
            elif G_sub.vs[sub_id].outdegree() == 1: role = 'bottom'
            else: print('Are you ok,', motif_name, '?')
          elif motif_name == 'S5':
            if G_sub.vs[sub_id].outdegree() == 2: role = 'bottom'
            elif G_sub.vs[sub_id].indegree() == 1: role = 'top'
            else: print('Are you ok,', motif_name, '?')
          else: role = 'none'
          name = G_sub.vs[sub_id]['name']
          row = (graph_name, node_id, name, motif_name, role)
          rows.append(row)

  columns = ['graph_name', 'node_id', 'node_name', 'motif_name', 'role']
  return pd.DataFrame(rows, columns = columns)

def process_row(i, G):
    motifs_name = list(load_motifs().keys())

    row_tc_G_swap = []
    h_c_swap = motif_contained_in_G_fast(swap_G_v2(G))
    row_tc_G_swap.append(f"{i:03d}")
    row_tc_G_swap.append(G['name'])
    for motif in motifs_name:
        row_tc_G_swap.append(h_c_swap[motif])

    return tuple(row_tc_G_swap)