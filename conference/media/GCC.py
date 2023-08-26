#%%
import igraph as ig
import matplotlib.pyplot as plt

edges_list = [[2,3], [3,4], [4,2],
         [0,2], [1,3],
         [4,5], [4,6]]

G = ig.Graph(n = 7, edges=edges_list)

fig, ax = plt.subplots()

ig.plot(G, ax=ax)
fig.show()