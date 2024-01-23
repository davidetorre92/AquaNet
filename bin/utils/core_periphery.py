import numpy as np
import pandas as pd

import igraph as ig
import itertools

def get_core_periphery_structure(G):
    # Get the largest strongly connected component
    sccs = G.components(mode=ig.STRONG)
    max_scc = max(sccs, key=len)
    # Convert to set for efficiency
    max_scc = set(max_scc)
    # Compute the IN and OUT sets
    ## Compute the nodes that reaches each vertex of the SCC
    reachable_in = G.neighborhood(vertices=max_scc, order=G.vcount(), mode='in', mindist=0)
    ## Convert to a set for efficiency
    reachable_in = set(itertools.chain.from_iterable(reachable_in))

    ## Compute the nodes that are reached by each vertex of the SCC
    reachable_out = G.neighborhood(vertices=max_scc, order=G.vcount(), mode='out', mindist=0)
    ## Convert to a set for efficiency
    reachable_out = set(itertools.chain.from_iterable(reachable_out))

    ## Finally compute the sets
    in_set = reachable_in - max_scc
    out_set = reachable_out - max_scc

    # Compute the Tubes, Tendrils IN sets
    tubes = set()
    tendrils_in = set()

    in_neighbors = G.neighborhood(vertices=in_set, order=G.vcount(), mode='out', mindist=0)
    in_neighbors = set(itertools.chain.from_iterable(in_neighbors))
    neighbors_to_be_tested = in_neighbors - max_scc - out_set - in_set

    for n in neighbors_to_be_tested:
        out_neighbors_n = set(G.neighborhood(vertices=n, order=G.vcount(), mode='out', mindist=0))
        if len(out_neighbors_n.intersection(out_set)) == 0:
            tendrils_in.add(n)
        else:
            tubes.add(n)

    # Compute the Tendrils OUT and the Disconnected sets
    tendrils_out = set()
    disconnected = set()

    all_others_nodes = set([v.index for v in G.vs()]) - max_scc - in_set - out_set - tendrils_in - tubes

    for n in all_others_nodes:
        out_neighbors_n = set(G.neighborhood(vertices=n, order=G.vcount(), mode='out', mindist=0))
        if len(out_neighbors_n.intersection(out_set)) == 0:
            disconnected.add(n)
        else:
            tendrils_out.add(n)

    sets = {
        'Core': max_scc,
        'IN set': in_set,
        'OUT set': out_set,
        'Tubes': tubes,
        'Tendrils IN': tendrils_in,
        'Tendrils OUT': tendrils_out,
        'Disconnected set': disconnected
    }

    return sets


def get_core_periphery_measurements(G_dataset, frac=True):
  names_column = []
  core, in_set, out_set, tubes, tendrils_in, tendrils_out, disconnected = [], [], [], [], [], [], []
  for G in G_dataset:
    V = G.vcount()
    names_column.append(G['name'])
    components = get_core_periphery_structure(G)
    core.append(len(components['Core']))
    in_set.append(len(components['IN set']))
    out_set.append(len(components['OUT set']))
    tubes.append(len(components['Tubes']))
    tendrils_in.append(len(components['Tendrils IN']))
    tendrils_out.append(len(components['Tendrils OUT']))
    disconnected.append(len(components['Disconnected set']))
    if frac:
      core[-1] = core[-1] / V
      in_set[-1] = in_set[-1] / V
      out_set[-1] = out_set[-1] / V
      tubes[-1] = tubes[-1] / V
      tendrils_in[-1] = tendrils_in[-1] / V
      tendrils_out[-1] = tendrils_out[-1] / V
      disconnected[-1] = disconnected[-1] / V

  bow_tie = pd.DataFrame({
                            'Core': core,
                            'In': in_set,
                            'Out': out_set,
                            'T-In': tendrils_in,
                            'T-out': tendrils_out,
                            'Tubes': tubes,
                            'Disconnected': disconnected
                                    }, index=names_column)
  return bow_tie