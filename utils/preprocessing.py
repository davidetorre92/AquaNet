import numpy as np
from utils.measurements import get_weighted_in_degree_and_adjacency_matrix, get_in_degree_and_adjacency_matrix
import re

# Node conversion dictionary to handle similar node names across datasets
node_conversion_dictionary = {'Adult': 'Large', 
            'Juvenile horse': 'Small Horse',
            'Small Horse mac': 'Small Horse',
            'Flatfish 1': 'Lemon Sole',
            'Flatfish 2': 'Large Flatfish',
            'toothed mammals': 'Toothed cetaceans',
            'Ad\.hmack.': 'Large horse mac',
            'Apexchond': 'Apex chondricht',
            'Chubmackerel': 'Chub mackerel',
            'Juv\.hmack\.': 'Small horse',
            'LargeM.capens': 'Large M. capens',
            'LargeM.parad': 'Large M. parado',
            'Otherlargepel': 'Other large pelagics',
            'Othersmallpel': 'Other small pelagics',
            'SmallM.capens': 'Small M. capens',
            'SmallM.parad': 'Small M. parado',
            'Benth.producers': 'Benthic produce',
            'Chondrichthyans': 'Chondrichtians',
            'Hake juvenile': 'Hake juv',
            'Gelatinous zoo.': 'Gelatinous zoop',
            'Microzooplank\.': 'Microzooplankto',
            'Mesozooplank\.': 'Mesozooplankton',
            'Macrozooplank\.': 'Macrozooplankto',
            'Othersmallpel': 'Other small pel',
            'Otherlargepel': 'Other large pel',
            'Living SED': 'Living sediment',
            'Float Veg.': 'Float. vegetation',
            'Periphyton\/Macroalgae': 'Periphyton',
            'Vine Leaves': 'Vine L',
            'Hardwood L': 'Hardwoods Leaves',
            'Cypress Leaves': 'Cypress L',
            'Cypress Wood': 'Cypress W',
            'HW Wood': 'Hardwood W',
            'Aquatic Invertebrates': 'Aquatic I',
            'Terrst. I': 'Ter. Invertebrates',
            'Fish HO': 'Small Fish, herb + omniv',
            'Fish PC': 'Small Fish, prim. carniv',
            'L Fish': 'Large Fish',
            'L Frog': 'Large Frogs',
            'M Frog': 'Medium Frogs',
            'S Frog': 'Small Frogs',
            'Salam. L': 'Salamander L',
            'GB Heron': 'Great blue heron',
            'Vert. det': 'Vertebrate Det.',
            'Benthic-feedi23': 'Benthicdemers',
            'Benthic-feeding': 'Benthicchond',
            'Pelagic-feedi22': 'Pelagicdemers',
            'Pelagic-feeding': 'Pelagicchond'
            }

def get_subgraph_living_creature(G, mode='rem_isolated'):
    living_nodes = [v for v in G.vs if v['ECO'] == 1]
    G_liv = G.subgraph(living_nodes)
    if mode == 'rem_isolated':
        G_liv = G_liv.subgraph([v for v in G_liv.vs if v.indegree() + v.outdegree() > 0 ])
    elif mode == 'keep_all':
        pass
    else:
        raise ValueError(f'Mode {mode} not implemented')

    return G_liv

def get_trophic_levels(G, epsilon = 1e-10, niter = 1000, weight = True):

  if weight:
    k_minus_1, I = get_weighted_in_degree_and_adjacency_matrix(G)
  else:
    k_minus_1, I = get_in_degree_and_adjacency_matrix(G)

  A = I.T
  V = k_minus_1.shape[0]
  s_t = np.ones(V)
  iter = 0
  while iter < niter:
    s_t_plus_1 = np.multiply(k_minus_1, np.matmul(A, s_t)) + np.ones(V)
    iter += 1
    if np.linalg.norm(s_t - s_t_plus_1) < epsilon:
      break
    else:
      s_t = s_t_plus_1.copy()
  return s_t

def map_nodes_to_code(name):
    # Map key words e.g: 'Adult' -> 'Large', 'Juvenile' -> 'Small'
    for k, v in node_conversion_dictionary.items():
        name = re.sub(k, v, name)
    # Capitalize first letter of each word
    name = re.sub(r'[^\s\w]', '', name)
    # Remove double spaces and spaces at the beginning and the end
    name = re.sub(r'\s+', ' ', name).strip()
    # Capitalize first letter of each word
    name = name.title()
    # Take the alternate letters
    return name

def nodes_difference_analysis(G_1, G_2):
    name_1 = G_1['name']
    name_2 = G_2['name']
    nodes_to_code_1 = {name: map_nodes_to_code(name) for name in G_1.vs['name']}
    nodes_to_code_2 = {name: map_nodes_to_code(name) for name in G_2.vs['name']}
    nodes_1 = set([nodes_to_code_1[v['name']] for v in G_1.vs])
    nodes_2 = set([nodes_to_code_2[v['name']] for v in G_2.vs])
    nodes_diff = nodes_1 ^ nodes_2
    result_line = []
    # If differences have been found, output_lines.append them
    result_line.append("\t\\item[\\textbf{Nodes}:] ")
    if len(nodes_diff) > 0:
        result_line.append(f"{len(nodes_diff)} nodes difference from a total of {len(nodes_1)} and {len(nodes_2)} respectively.\n")
        node_diff_1 = [v['name'] for v in G_1.vs if nodes_to_code_1[v['name']] in nodes_diff]
        node_diff_2 = [v['name'] for v in G_2.vs if nodes_to_code_2[v['name']] in nodes_diff]
        result_line.append("\n\t\t\\begin{itemize}\n")
        result_line.append(f"\t\t\t\item[{name_1}]: {node_diff_1}\n\t\t\t\item[{name_2}]: {node_diff_2}\n")
        result_line.append("\t\t\end{itemize}\n")
    else:
        result_line.append(f"No nodes differences\n")
    return result_line

def edges_difference_analysis(G_1, G_2):
    name_1 = G_1['name']
    name_2 = G_2['name']
    nodes_to_code_1 = {name: map_nodes_to_code(name) for name in G_1.vs['name']}
    nodes_to_code_2 = {name: map_nodes_to_code(name) for name in G_2.vs['name']}
    # output_lines.append edges difference
    edges_1_names = [(G_1.vs['name'][e.tuple[0]], G_1.vs['name'][e.tuple[1]]) for e in G_1.es]
    edges_2_names = [(G_2.vs['name'][e.tuple[0]], G_2.vs['name'][e.tuple[1]]) for e in G_2.es]
    edges_1_code = set([(nodes_to_code_1[e[0]], nodes_to_code_1[e[1]]) for e in edges_1_names])
    edges_2_code = set([(nodes_to_code_2[e[0]], nodes_to_code_2[e[1]]) for e in edges_2_names])
    edges_diff = edges_1_code ^ edges_2_code
    result_line = []
    result_line.append("\t\\item[\\textbf{Edges}:] ")

    # If differences have been found, output_lines.append them
    if len(edges_diff) > 0:
        result_line.append(f"{len(edges_diff)} edges difference out of {len(edges_1_code)} and {len(edges_2_code)} respectively\n")
        edge_diff_1 = [e for e in edges_1_names if (nodes_to_code_1[e[0]], nodes_to_code_1[e[1]]) in edges_diff]
        edge_diff_2 = [e for e in edges_2_names if (nodes_to_code_2[e[0]], nodes_to_code_2[e[1]]) in edges_diff]
        result_line.append("\n\t\t\\begin{itemize}\n")
        result_line.append(f"\t\t\t\item[{name_1}]: {edge_diff_1}\n\t\t\t\item[{name_2}]: {edge_diff_2}\n")
        result_line.append("\t\t\end{itemize}\n")
    else:
        result_line.append(f"No edges differences\n")
    return result_line

def core_periphery_difference_analysis(G_1, G_2, df):
    name_1 = G_1['name']
    name_2 = G_2['name']
    nodes_to_code_1 = {name: map_nodes_to_code(name) for name in G_1.vs['name']}
    nodes_to_code_2 = {name: map_nodes_to_code(name) for name in G_2.vs['name']}

    # Core and periphery differences
    df_G_1 = df[df['Network'] == name_1]
    df_G_2 = df[df['Network'] == name_2]
    species_core_periphery_1 = set([(nodes_to_code_1[name], struct) for name, struct in zip(df_G_1['Node name'], df_G_1['Core periphery'])])
    species_core_periphery_2 = set([(nodes_to_code_2[name], struct) for name, struct in zip(df_G_2['Node name'], df_G_2['Core periphery'])])
    species_core_periphery_diff = species_core_periphery_1 ^ species_core_periphery_2
    result_line = []
    result_line.append("\t\\item[\\textbf{Core periphery}:] ")

    if len(species_core_periphery_diff) > 0:
        species_core_periphery_diff_1 = [(name, struct) for name, struct in species_core_periphery_1 if (name, struct) in species_core_periphery_diff]
        species_core_periphery_diff_2 = [(name, struct) for name, struct in species_core_periphery_2 if (name, struct) in species_core_periphery_diff]
        result_line.append("\n\t\t\\begin{itemize}\n")
        result_line.append(f"\t\t\t\item[{name_1}]: {species_core_periphery_diff_1}\n\t\t\t\item[{name_2}]: {species_core_periphery_diff_2}\n")
        result_line.append("\t\t\end{itemize}\n")
    else:
        result_line.append("No core-periphery differences\n")
    result_line.append("\n")
    return result_line
