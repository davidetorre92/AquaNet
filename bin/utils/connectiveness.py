from itertools import combinations
import numpy as np
import pandas as pd

import igraph as ig
import networkx as nx

def n_reachable_pairs(G):

  G_nx = G.to_networkx()
  G_close = nx.transitive_closure(G_nx)

  return G_close.number_of_edges()

def greedy_G_v(G, v, function=n_reachable_pairs):
  """
  Greedy strategy function:
  G is the graph
  v is the vertex to be tested
  f_G is the connectivity funciton measured AFTER the removal of from G.
  """
  G_temp = G.copy()
  G_temp.delete_vertices(v)
  return function(G_temp)

def min_index(nodes, f_G_del, all=False):
  """
  This function sorts the nodes according to the measure f_G_del made upon the strategy
  and the indices of the vertex, so that if more than one vertex has the same f_G_del, then one with
  the least index is selected.
  """
  experiment = zip(nodes, f_G_del)
  experiment = sorted(experiment, key = lambda x: x[0].index)
  experiment = sorted(experiment, key = lambda x: x[1])
  if all: selected_node_s = [el[0] for el in experiment]
  else: selected_node_s = experiment[0][0]
  return selected_node_s

def node_removal_strategy(G,
                          measure_v = lambda G, v: greedy_G_v(G, v),
                          vertices_collecting_function = lambda G: [v for v in G.vs() if v['ECO'] == 1],
                          vertices_ordering_function = lambda nodes, f_G_del: min_index(nodes, f_G_del),
                          stop_condition = lambda G: len([v for v in G.vs() if v['ECO'] == 1]) > 0
                          ):
  """
  This function takes a graph and a strategy and returns the list of vertices that minimize the
  connectivity function according to the strategy.
  A strategy is composed by these elements:
  1. a measure on the vertices;
  2. a function that collects the vertices from the graph given the properties of the vertex;
  3. a function that sorts and collects the vertices given the measure;
  4. a stop condition that stops the loop on the graph;

  With a combination of these 4 elements it is possible to test different strategies.
  For example, one may test how does it change the connectivity value of the graph
  when the vertices that minimize the most the connectivity value are removed first.
  In this case, the measure on the vertices is the connectivity function given the graph after the removal of a vertex;
  the function that collects the vertices is a function that takes all vertices of the graph regardless their properties;
  the sorting and collecting function sorts the vertices according to the measured connectivity function after their removal and collect
  the vertex that minimize the most the connectivity; the loop stop when there are no more vertices in the graph, i.e. when the graph is empty.
  """

  # 1. List to be returned: the nodes given the ordering and the function
  list_of_removed_nodes = []
  # 2. G.delete_vertices in an inplace function, therefore a dummy variable is needed
  G_M = G.copy()
  # 3. Loop on the graph given a condition
  while(stop_condition(G_M)):
    # 3.1 Collect the nodes with a fixed order
    v_order = vertices_collecting_function(G_M)
    # 3.2 Measure the vertices property given by measure_v
    f_G_del = [measure_v(G_M, v) for v in v_order]
    # 3.3 Sort and select the vertices
    selected_v_s = vertices_ordering_function(v_order, f_G_del)
    # 3.4 Handling different strategies: vertices_ordering_function can return a single vertex of a list of vertices
    if type(selected_v_s) == list:
      list_of_removed_nodes = [v['name'] for v in selected_v_s]
    else:
      list_of_removed_nodes.append(selected_v_s['name'])
  #Node(s) is removed to meet the stop condition
    G_M.delete_vertices(selected_v_s)
  return list_of_removed_nodes

def get_reachable_pairs_change(G, list_of_removed_nodes):
  graph_name = G['name']
  rows_G = []
  G_temp = G.copy()
  V = G.vcount()
  total_pairs = V ** 2

  r_p_G = n_reachable_pairs(G_temp)
  rows_G.append((0, graph_name, 'No nodes removed', r_p_G, r_p_G / total_pairs))
  for i_node, vertex in enumerate(list_of_removed_nodes):
    G_temp.delete_vertices(vertex)
    node_fraction = (i_node + 1) / V
    r_p_G = n_reachable_pairs(G_temp)
    rows_G.append((node_fraction, graph_name, vertex, r_p_G, r_p_G / total_pairs))

  return pd.DataFrame(rows_G, columns=['node_fraction', 'graph_name', 'node_removed', 'n_reachable_pairs', 'percentage_reachable_pairs'])

def robustness_function_reachable_pairs(df_node_removal_G):
  values = df_node_removal_G['percentage_reachable_pairs'].to_numpy()
  N = df_node_removal_G.shape[0]
  return np.sum(values) / N

def robustness_function(df_node_removal_G):
  values = df_node_removal_G['f_G / f_G_max'].to_numpy()
  N = df_node_removal_G.index[-1]
  return np.sum(values) / N

def robustness_function_new(df_node_removal_G):
  values = df_node_removal_G['f_G / f_G_fully'].to_numpy()
  N = df_node_removal_G.shape[0] - 1
  return np.sum(values) / N