#%%
import igraph as ig
import matplotlib.pyplot as plt

edges_list = [[2,3], [3,4], [4,2],
         [0,2], [1,3],
         [4,5], [4,6]]

G = ig.Graph(n = 7, edges=edges_list, directed = True)
layout = G.layout(layout='auto')
visual_style = {}

fig, ax = plt.subplots()
visual_style['vertex_color'] = ['white' for _ in range(7)]
ig.plot(G, target=ax, layout=layout, **visual_style)
fig.savefig("graph_1.pdf")

fig, ax = plt.subplots()
visual_style['vertex_color'][2] = '#440154'
visual_style['vertex_color'][3] = '#440154'
visual_style['vertex_color'][4] = '#440154'
ig.plot(G, target=ax, layout=layout, **visual_style)
fig.savefig("graph_2.pdf")

fig, ax = plt.subplots()
visual_style['vertex_color'][0] = '#2A788E'
visual_style['vertex_color'][1] = '#2A788E'
ig.plot(G, target=ax, layout=layout, **visual_style)
fig.savefig("graph_3.pdf")

fig, ax = plt.subplots()
visual_style['vertex_color'][5] = '#7AD151'
visual_style['vertex_color'][6] = '#7AD151'
ig.plot(G, target=ax, layout=layout, **visual_style)
fig.savefig("graph_4.pdf")