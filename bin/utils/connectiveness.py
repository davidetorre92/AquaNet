import numpy as np
import igraph as ig

def f_G(G):
  C = G.components(mode='strong')
  C_card = [len(c_i) for c_i in C]
  return np.sum(np.array([C_i * (C_i - 1) / 2 for C_i in C_card]))