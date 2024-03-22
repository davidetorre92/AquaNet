import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import igraph as ig

tableu_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def visual_style(g, color_attribute = 'ECO', vertex_label_attribute = 'name', layout=None, size_mode=None):
  # set layout
  if layout is None:
    layout = g.layout_fruchterman_reingold(niter=1000)

  # these properties will be used to set the graph aspect
  try:
    weights = [e['weight'] for e in g.es()]
  except:
    weights = [1.0 for e in g.es()]

  visual_style = {}
  # set visual style
  visual_style = {}
  ## set vertex labels to be the name of the vertex of the graph
  if vertex_label_attribute is not None: visual_style['vertex_label'] = [v[vertex_label_attribute] for v in g.vs()]
  ## change label size according to the betweenness
  visual_style['vertex_size'] = 30
  ## change color according to the vertex color_attribute property

  if color_attribute is not None:
    palette = {
    0: '#1f77b4',
    1: '#ff7f0e',
    2: '#aec7e8',
    3: '#ffbb78',
    4: '#2ca02c',
    5: '#98df8a',
    6: '#d62728',
    7: '#ff9896',
    8: '#9467bd',
    9: '#c5b0d5',
    10: '#8c564b',
    11: '#c49c94',
    12: '#e377c2',
    13: '#f7b6d2',
    14: '#7f7f7f',
    15: '#c7c7c7',
    16: '#bcbd22',
    17: '#dbdb8d',
    18: '#17becf',
    19: '#9edae5'
    }

    n_palette = len(palette.keys())
    unique_attributes_color = set([v[color_attribute] for v in g.vs()])
    index_to_attribute = {attribute: index % n_palette for index, attribute in zip(range(len(unique_attributes_color)), unique_attributes_color)}

    visual_style['vertex_color'] = [palette[index_to_attribute[v[color_attribute]]] for v in g.vs()]
  else:
    visual_style['vertex_color'] = ['#1f77b4' for v in g.vs()]
  ## change the width of the verteces
  visual_style['vertex_frame_width'] = 1


  ## set edge color to be gray. The higher the weight of the lower the transparancy
  visual_style['edge_color'] = [(0.5, 0.5, 0.5, 1) for w in weights]
  ## change edge width
  visual_style['edge_width'] = 0.1


  ## change vertex labels' size
  visual_style['vertex_label_size'] = 10

  ## set the layout into the dictionary
  visual_style['layout'] = layout

  return visual_style

def ego_net(G, node, neighbor_order = 1, mode='all'):
  if type(node) == str:
    if node in G.vs['name'] is False:
      print(f'Node {node} not found. Aborting.')
      return False
    for v in G.vs:
      if node == v['name']: node = v
  elif type(node) == int:
    if node in [v.index for v in G.vs] is False:
      print(f'Node {node} not found. Aborting.')
      return False
    for v in G.vs:
      if node == v.index: node = v
  elif type(node) == ig.Vertex:
    if node in [v for v in G.vs] is False:
      print(f'Node {node} not found. Aborting.')
      return False
      node = v
  else:
    print('Node should be either a string, an integer or a ig.Vertex.')
    return False

  neighborhood = G.neighborhood(node, order = neighbor_order, mode = mode)
  v_subgraph = [node.index] + neighborhood
  G_sub = G.subgraph(v_subgraph)
  vs = visual_style(G_sub)
  vs['vertex_shape'] = ['circle' for _ in range(len(v_subgraph))]
  return G_sub, vs

def plot_graph(G, color_attribute = 'ECO', vertex_label_attribute = 'name', target = None):
    print(color_attribute)
    vs = visual_style(G, color_attribute = color_attribute, vertex_label_attribute = vertex_label_attribute)
    if target is None:
        fig, ax = plt.subplots()
    else:
        ax = target
        fig = ax.get_figure()

    ig.plot(G, target = ax, **vs)

    # Setting the handles
    color_label = set()
    labels = G.vs[color_attribute]
    colors = vs['vertex_color']
    for color, label in zip(colors, labels):
        color_label.add((color, f'{color_attribute}: {label}'))

    handles = [mpatches.Circle((0.5, 0.5), 0.5, facecolor = color, label = label) for color, label in color_label]
    handles = handles + [mpatches.Circle((0.0, 0.0), 0.0, color = 'white')]
    ax.legend(handles = handles, bbox_to_anchor=(1.0, 0.05))
    return fig, ax

def boxplot_percentages_and_print_data(df, ax = None):
  df = df * 100
  df_melted = df.melt(var_name='Category', value_name = 'Percentage')
  df_melted.fillna(0, inplace = True)
  fig, ax = boxplot_melt_data(df_melted, ax, var_name='Category', value_name = 'Percentage')
  return fig, ax

def boxplot_melt_data(df_melted, ax = None, var_name='Category', value_name = 'Percentage', 
                      palette = tableu_colors, median_color = '#000000', average_marker = '+'
):
  if ax == None:
    # Create a boxplot
    fig, ax = plt.subplots()
  else:
     fig = ax.get_figure()
  # Extract unique categories
  categories = df_melted[var_name].unique()
  # Prepare data for each category
  data_to_plot = [df_melted[df_melted[var_name] == category][value_name] for category in categories]

  # Custom colors
  
  colors = palette[:len(categories)]
  median_color = '#000000'
  average_marker = '+'
  # Creating the boxplot with custom fill colors
  bp = ax.boxplot(data_to_plot, patch_artist=True, showmeans = False, medianprops={'color': median_color})

  for patch, color in zip(bp['boxes'], colors):
      patch.set_facecolor(color)

  # Add '+' marker at the average position for each box
  for i, line in enumerate(data_to_plot):
      # Calculate average
      average = np.mean(line)
      # Plot average marker
      ax.plot(i + 1, average, average_marker, color='black')

  # Customize ticks on the x-axis
  ax.set_xticks(range(1, len(categories) + 1))
  ax.set_xticklabels(categories)

  # Setting y-axis to display percentage
  if value_name == 'Percentage': ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x}%'))

  # Print data:
  perc_str = '%' if value_name == 'Percentage' else ''
  for category in categories:
      data = df_melted[df_melted[var_name] == category][value_name]
      mean = np.mean(data)
      print(f'Average {category}: {mean:.3f}' + perc_str)
  return fig, ax