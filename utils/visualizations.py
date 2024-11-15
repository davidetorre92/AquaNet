import matplotlib.pyplot as plt
import numpy as np
import igraph as ig

def labelize(name):
    # splitted = name.split(' ')
    # litterals = ''.join([splitted[i] if splitted[i][0].isalpha() else '' for i in range(len(splitted))])
    # caps_lock_name = litterals.upper()[:3]
    return name
def get_position_on_grid_pyvis(G, bin_size = 0.3, empty_spaces = 4, vertex_radius = 10):
    empty_size_horizontal = empty_spaces + 1
    empty_size_vertical = 3
    vertex_diameter = vertex_radius * 2
    #1. Make the grid
    max_tl = max(G.vs['trophic_level'])
    min_tl = min(G.vs['trophic_level'])
    n_rows = int((max_tl - min_tl) / bin_size) + 1
    node_to_row = [int((v['trophic_level'] - min_tl) / bin_size) for v in G.vs]
    available_rows, cols_per_row = np.unique(node_to_row, return_counts = True)
    pos_layout = np.array([[x,y] for x,y in G.layout('kk')]) # not sure if pos_layout[i] corresponds to the position of ith node or something else...
    x_min = np.min(pos_layout[:,0])
    x_max = np.max(pos_layout[:,0])
    pos_layout_norm = (pos_layout[:,0] - x_min) / (x_max - x_min)
    col_positions_norm = {row: np.array([icol / cols_per_row[irow] for icol in range(cols_per_row[irow])]) for irow, row in zip(range(len(cols_per_row)), available_rows)}
    
    pos = np.zeros((G.vcount(), 2))
    for inode in range(G.vcount()):
        row = node_to_row[inode]
        closest_col = np.argmin(np.abs(col_positions_norm[row] - pos_layout_norm[inode]))
        pos[inode][0] = (col_positions_norm[row][closest_col] * vertex_diameter * empty_size_horizontal) * (x_max - x_min) + x_min
        pos[inode][1] = - ((row - n_rows // 2) * vertex_diameter * empty_size_vertical)
        col_positions_norm[row][closest_col] = np.inf
    return pos

def get_position_on_grid(G, bin_size = 0.3, empty_spaces = 4, vertex_radius = 10):
    empty_size_horizontal = empty_spaces + 1
    empty_size_vertical = 3
    vertex_diameter = vertex_radius * 2
    #1. Make the grid
    max_tl = max(G.vs['trophic_level'])
    min_tl = min(G.vs['trophic_level'])
    n_rows = int((max_tl - min_tl) / bin_size) + 1
    node_to_row = [int((v['trophic_level'] - min_tl) / bin_size) for v in G.vs]
    available_rows, cols_per_row = np.unique(node_to_row, return_counts = True)
    pos_layout = np.array([[x,y] for x,y in G.layout('kk')]) # not sure if pos_layout[i] corresponds to the position of ith node or something else...
    x_min = np.min(pos_layout[:,0])
    x_max = np.max(pos_layout[:,0])
    pos_layout_norm = (pos_layout[:,0] - x_min) / (x_max - x_min)
    col_positions_norm = {row: np.array([icol / cols_per_row[irow] for icol in range(cols_per_row[irow])]) for irow, row in zip(range(len(cols_per_row)), available_rows)}
    
    pos = np.zeros((G.vcount(), 2))
    for inode in range(G.vcount()):
        row = node_to_row[inode]
        closest_col = np.argmin(np.abs(col_positions_norm[row] - pos_layout_norm[inode]))
        pos[inode][0] = (col_positions_norm[row][closest_col] * vertex_diameter * empty_size_horizontal) * (x_max - x_min) + x_min
        pos[inode][1] = ((row - n_rows // 2) * vertex_diameter * empty_size_vertical) * 100
        col_positions_norm[row][closest_col] = np.inf
    # min_y_plot = np.min(pos[:,1])
    # max_y_plot = np.max(pos[:,1])
    # min_tl = np.min(G.vs['trophic_level'])
    # max_tl = np.max(G.vs['trophic_level'])
    # tl_min_max = [(v['trophic_level'] - min_tl) / (max_tl - min_tl) for v in G.vs]
    # pos[:,1] = [tl * (max_y_plot - min_y_plot) + min_y_plot for tl in tl_min_max]
    return pos

def plot_trophic_level(G, highlight_nodes = None, ax = None):
    edge_color_dictionary = {'11': '#0055aa',
                  '22': '#115500',
                  '12': '#111111',
                  '21': '#990099'}
    vertex_color_dictionary = {'1': '#00025b',
                               '2': '#ff8200'}
    print()
    if ax is None:
        fig, ax = plt.subplots(figsize=(9,9))
    else:
        fig = ax.get_figure()

    layout = get_position_on_grid(G)
    labels = [name if name in highlight_nodes else '' for name in G.vs['name']]
    vertex_color = [vertex_color_dictionary[f"{int(v['ECO'])}"] for v in G.vs]
    edge_color = [edge_color_dictionary[f"{int(G.vs[e.source]['ECO'])}{int(G.vs[e.target]['ECO'])}"] for e in G.es]
    if highlight_nodes != []: 
        vertex_color = [vc + 'ff' if name in highlight_nodes else vc + '22' for vc, name in zip(vertex_color, G.vs['name'])]
        edge_color = [ec + 'ff' if (G.vs[edge.source]['name'] in highlight_nodes) and (G.vs[edge.target]['name'] in highlight_nodes) else ec + '05' for ec, edge in zip(edge_color, G.es)]
        edge_width = [2 if (G.vs[edge.source]['name'] in highlight_nodes) and (G.vs[edge.target]['name'] in highlight_nodes) else 0.5 for ec, edge in zip(edge_color, G.es)]
    else:
        vertex_color = [vc + 'ff' for vc in vertex_color]
        edge_color = [ec + '15' for ec in edge_color]
        edge_width = 2
    # Plot graph
    ig.plot(G, #graph = G,
            layout = layout, # layout = G.layout('kk'),
            target = ax, # axes = ax,
            vertex_label = labels, # vertex_label = G.vs['name'] if the label is the nodes to be highlighted
            vertex_label_size = 10,
            vertex_label_color = 'black',
            edge_width = edge_width,
            vertex_size = 10,
            vertex_color = vertex_color,
            edge_color = edge_color,  # Set edge color to red
            edge_arrow_size = 5,
            edge_arrow_width = 5)
    ax.set_axis_off()
    return fig, ax

def get_vertex_names(G):
    vertex_names = [v['name'].replace(' ', '\n') for v in G.vs]
    return vertex_names

def plot_foodwebs_by_trophic_levels(G, ax = None, ego = None, coarse = 0.01, vertex_size = 20):
    if ax is None:
        fig, ax = plt.subplots()
    pos = G.layout('kk') # initalize null layout

    n_nodes = len(pos)
    max_trophic_level = max(G.vs['trophic_level'])
    min_trophic_level = min(G.vs['trophic_level'])
    n_rows = int(np.ceil((max_trophic_level - min_trophic_level) / coarse)) + 1
    n_rows = max_trophic_level - min_trophic_level + 1
    n_cols = int(np.ceil(n_nodes / n_rows))
    x_pos = [int(G.vs[inode]['trophic_level'] - min_trophic_level) + (inode) * (vertex_size + 1) for inode in range(n_nodes)]
    y_pos = [int(inode / n_cols) * (vertex_size + 1) for inode in range(n_nodes)]
    for inode in range(len(pos)):
        x = x_pos[inode]
        y = y_pos[inode]
        pos[inode] = (x, y)
        y_pos = [int(G.vs[inode]['trophic_level'] / coarse + coarse * 0.5) * coarse for inode in range(len(pos))]

    vertex_names = get_vertex_names(G)
    vertex_color = ['blue' if v['ECO'] == 1 else 'orange' for v in G.vs]
    if ego is not None:
        for i in range(len(vertex_names)):
            if vertex_names[i] == ego:
                break
        vertex_color[i] = 'red'
        
    ig.plot(G, layout = pos, target = ax, 
            vertex_label = vertex_names, 
            vertex_color = vertex_color, 
            vertex_size = vertex_size, 
            edge_width = 0.5,
            vertex_label_size = 8,
            awes=0.5,
            edge_arrow_size = 5, edge_arrow_width = 5)
    return ax

def plot_subgraph(G, nodes, trophic_level = True, ax = None):
    if ax is None:
        fig, ax = plt.subplots()
    
    subgraph = G.subgraph(nodes)
    if trophic_level:
        ax = plot_foodwebs_by_trophic_levels(subgraph, ax = ax)
    else:
        pos = subgraph.layout('kk')
        vertex_names = subgraph.vs['name']
        vertex_color = ['blue' if v['ECO'] == 1 else 'orange' for v in subgraph.vs]
    ig.plot(G, layout = pos, target = ax, 
            vertex_label = vertex_names, 
            vertex_color = vertex_color, 
            vertex_size = 20, 
            edge_width = 0.5,
            vertex_label_size = 8,
            awes=0.5,
            edge_arrow_size = 5, edge_arrow_width = 5)
    return ax

def plot_motif(G, nodes, motif_name, ax = None):
    subgraph = G.subgraph(nodes)
    if ax is None:
        fig, ax = plt.subplots()
    
    if motif_name == 'S2':
        xs = [0, 1, -1]
    elif motif_name == 'S4':
        xs = [1, -1, 0]
    elif motif_name == 'S5':
        xs = [0, 1, -1]
    
    nodes = sorted([v for v in subgraph.vs], key = lambda v: v['trophic_level'])
    pos = [(x, v['trophic_level']) for x, v in zip(xs, nodes)]

    vertex_names = [v['name'].replace(' ', '\n') for v in nodes]
    vertex_color = ['blue' if v['ECO'] == 1 else 'orange' for v in subgraph.vs]
    ig.plot(subgraph, layout = pos, target = ax, 
            vertex_label = vertex_names, 
            vertex_color = vertex_color, 
            vertex_size = 20, 
            edge_width = 0.5,
            vertex_label_size = 8,
            awes=0.5,
            edge_arrow_size = 5, edge_arrow_width = 5)

    return ax