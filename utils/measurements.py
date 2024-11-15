from random import sample
import itertools
import numpy as np
import igraph as ig
import networkx as nx
import pdb

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
    set_test = [max_scc, in_set, out_set, tubes, tendrils_in, tendrils_out, disconnected]
    for i in range(len(set_test)):
        for j in range(i+1, len(set_test)):
            intersection = set_test[i].intersection(set_test[j])
            if len(intersection) > 0:
                print(f"Intersection found: {intersection} between {i} and {j}")
                raise ValueError
    return sets
def count_first_order_motifs(adjacency_matrix):
    # exclude cannibalism
    np.fill_diagonal(adjacency_matrix, 0)
    # count edges
    single_edges = np.argwhere((adjacency_matrix == 1) & (adjacency_matrix.T != 1))
    double_edges = np.argwhere((adjacency_matrix == 1) & (adjacency_matrix.T == 1)) 
    return single_edges.shape[0], double_edges.shape[0] // 2
def preprocess_adjacency_matrix_and_check_first_order_motifs(adjacency_matrix_og):
    adjacency_matrix = adjacency_matrix_og.copy()
    # # convert to binary
    adjacency_matrix = np.where(adjacency_matrix > 0, 1, 0)
    # exclude cannibalism
    np.fill_diagonal(adjacency_matrix, 0)
    # count edges
    E = np.sum(adjacency_matrix)
    E_single = np.sum((adjacency_matrix == 1) & (adjacency_matrix.T != 1))
    E_double = np.sum((adjacency_matrix == 1) & (adjacency_matrix.T == 1))
    if E_single + E_double != E:
        raise ValueError('Counting edge error')
    return adjacency_matrix

def swap_adj_matrix(adjacency_matrix_og, seed, swaps = 100):
    def get_edge_lists(adjacency_matrix_og):
        single_edge_list = np.argwhere((adjacency_matrix_og > 0) & (adjacency_matrix_og.T == 0))
        double_edge_list = np.argwhere((adjacency_matrix_og > 0) & (adjacency_matrix_og.T > 0))
        double_edge_list = double_edge_list[double_edge_list[:,0] < double_edge_list[:,1]]
        single_edge_list = single_edge_list.tolist()
        double_edge_list = double_edge_list.tolist()
        
        return single_edge_list, double_edge_list
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

    adjacency_matrix = preprocess_adjacency_matrix_and_check_first_order_motifs(adjacency_matrix_og)
    single_edge_list, double_edge_list = get_edge_lists(adjacency_matrix)
    E_single = len(single_edge_list)
    E_double = len(double_edge_list)

    if E_single >= 2:
        for _ in range(swaps * E_single):
            iA, iB = sample(range(E_single), 2)
            if is_valid_edge_swap(single_edge_list, double_edge_list, single_edge_list[iA], single_edge_list[iB]) is False: continue
            A, B = single_edge_list[iA]
            C, D = single_edge_list[iB]
            #swap
            single_edge_list[iA][1] = D
            single_edge_list[iB][1] = B

    if E_double >= 2:
        for n in range(swaps * E_double):
            iA, iB = sample(range(E_double), 2)
            if is_valid_edge_swap(single_edge_list, double_edge_list, double_edge_list[iA], double_edge_list[iB]) is False: continue
            A, B = double_edge_list[iA]
            C, D = double_edge_list[iB]
            #swap
            double_edge_list[iA][1] = D
            double_edge_list[iB][1] = B

    edge_list_swap = [[edge[0], edge[1]] for edge in single_edge_list] + [[edge[0], edge[1]] for edge in double_edge_list] + [[edge[1], edge[0]] for edge in double_edge_list]

    adjacency_matrix_swap = np.zeros_like(adjacency_matrix_og)
    for edge in edge_list_swap:
        adjacency_matrix_swap[edge[0], edge[1]] = 1
    return adjacency_matrix_swap

def randomize_network_keep_deg_sequence(adjacency_matrix_og, seed = None, swaps = 20):

    adjacency_matrix = preprocess_adjacency_matrix_and_check_first_order_motifs(adjacency_matrix_og)
    E_single = np.argwhere((adjacency_matrix == 1) & (adjacency_matrix.T != 1)).shape[0]
    E_double = np.argwhere((adjacency_matrix == 1) & (adjacency_matrix.T == 1)).shape[0]
    # randomize
    if seed is not None:
        np.random.seed(seed)
    for iteration in range(swaps * E_single):
        edge_list_single = np.argwhere((adjacency_matrix == 1) & (adjacency_matrix.T != 1))
        if len(edge_list_single) < 2:
            print(f"Break after {iteration} iterations")
            break # no more single edges
        a, b = edge_list_single[np.random.randint(len(edge_list_single))]
        selector_adjacency_matrix = get_selector_adjacency_matrix(adjacency_matrix, a, b, mode = 'single')
        edge_list_single = np.argwhere(selector_adjacency_matrix == 1)
        if len(edge_list_single) == 0:
            continue
        c, d = edge_list_single[np.random.randint(len(edge_list_single))]
        adjacency_matrix[a,b] = 0
        adjacency_matrix[c,d] = 0
        adjacency_matrix[a,d] = 1
        adjacency_matrix[c,b] = 1
        
    for iteration in range(swaps * E_double // 2):
        edge_list_double = np.argwhere((adjacency_matrix == 1) & (adjacency_matrix.T == 1))
        if len(edge_list_double) < 2:
            print(f"Break after {iteration} iterations")
            break # no more double edges
        a, b = edge_list_double[np.random.randint(len(edge_list_double))]
        selector_adjacency_matrix = get_selector_adjacency_matrix(adjacency_matrix, a, b, mode = 'double')
        edge_list_double = np.argwhere(selector_adjacency_matrix == 1)
        if len(edge_list_double) == 0:
            continue
        c, d = edge_list_double[np.random.randint(len(edge_list_double))]
        adjacency_matrix[a,b] = 0
        adjacency_matrix[b,a] = 0
        adjacency_matrix[c,d] = 0
        adjacency_matrix[d,c] = 0
        adjacency_matrix[a,d] = 1
        adjacency_matrix[d,a] = 1
        adjacency_matrix[c,b] = 1
        adjacency_matrix[b,c] = 1
    return adjacency_matrix

def randomize_network_directed_curveball(adjacency_matrix_og, seed = None, swaps = 100):
    pdb.set_trace()
    adjacency_matrix = preprocess_adjacency_matrix_and_check_first_order_motifs(adjacency_matrix_og)
    E_single = np.argwhere((adjacency_matrix == 1) & (adjacency_matrix.T != 1)).shape[0]
    E_double = np.argwhere((adjacency_matrix == 1) & (adjacency_matrix.T == 1)).shape[0]
    # randomize
    if seed is not None:
        np.random.seed(seed)
    nodes_indices = list(range(adjacency_matrix.shape[0]))
    for iteration in range(swaps * adjacency_matrix.shape[0]):
        # pick two nodes
        i, j = np.random.choice(nodes_indices, 2, replace = False)
        # pick the outgoing nodes set
        A_i = np.argwhere(adjacency_matrix[i] > 0)
        A_j = np.argwhere(adjacency_matrix[j] > 0)
        # pick the nodes in A_i that are not in A_j
        A_i_j = np.setdiff1d(A_i, np.append(A_j, j))
        A_j_i = np.setdiff1d(A_j, np.append(A_i, i))
        # exclude those nodes in A_i_j that have an outgoing edge in A_j to avoid double links
        A_i_j = np.setdiff1d(A_i_j, np.argwhere(adjacency_matrix[A_i_j][:,A_j] > 0))
        # exclude those nodes in A_j_i that have an outgoing edge in A_i double links
        A_j_i = np.setdiff1d(A_j_i, np.argwhere(adjacency_matrix[A_j_i][:,A_i] > 0))
        # Create B_i and B_j
        ## Create A_i_j \interction A_j_i
        intersection_i_j_and_j_i = np.intersect1d(A_i_j, A_j_i)
        if len(intersection_i_j_and_j_i) == 0:
            continue
        ## shuffle the intersection
        np.random.shuffle(intersection_i_j_and_j_i)
        ## remove A_i_j from A_i
        intersection_i = np.intersect1d(A_i_j, A_i)
        n_elements_i = len(intersection_i)
        B_i = np.setdiff1d(A_i, A_i_j)
        ## add the same elements of removed elements in A_i choosen randomly from A_i_j \intersection A_j_i
        elements_in_B_i = np.random.choice(intersection_i_j_and_j_i, n_elements_i, replace = False)
        B_i = np.append(B_i, elements_in_B_i)
        intersection_i_j_and_j_i = np.setdiff1d(intersection_i_j_and_j_i, elements_in_B_i)
        ## remove A_j_i from A_j
        intersection_j = np.intersect1d(A_j_i, A_j)
        n_elements_j = len(intersection_j)
        B_j = np.setdiff1d(A_j, A_j_i)
        ## add the remaining elements of removed elements in A_j choosen randomly from A_i_j \intersection A_j_i
        B_j = np.append(B_j, np.random.choice(intersection_i_j_and_j_i, n_elements_j, replace = False))

        # Update the adjacency matrix
        adjacency_matrix[A_i] = 0
        adjacency_matrix[A_j] = 0
        adjacency_matrix[B_i] = 1
        adjacency_matrix[B_j] = 1

    return adjacency_matrix

def process_randomization(adjacency_matrix, adj_rand_function = randomize_network_keep_deg_sequence, seed = None):
    adjacency_matrix_random = adj_rand_function(adjacency_matrix, seed)
    motif_census_rand = motif_contained_in_adj_fast(adjacency_matrix_random)
    single_edges, double_edges = count_first_order_motifs(adjacency_matrix_random)
    deg_seq_og = np.sum(adjacency_matrix, axis = 1)
    deg_seq_rand = np.sum(adjacency_matrix_random, axis = 1)
    if np.all(deg_seq_og == deg_seq_rand) is False:
        raise ValueError('Degree sequence not equal after randomization')
    # Return the results to collect them in the main thread
    return (motif_census_rand, single_edges, double_edges)

def get_selector_adjacency_matrix(A_og, a, b, mode = 'single'):
    def collect_neighbors(A, u, exclude = None, mode = 'all', return_mode = 'list'):
        if mode == 'all':
            neighbors = set(np.argwhere((A.T[:,u] == 1) | (A[:,u] == 1)).T[0])
        elif mode == 'out':
            neighbors = set(np.argwhere(A.T[:,u] == 1).T[0])
        elif mode == 'in':
            neighbors = set(np.argwhere(A[:,u] == 1).T[0])
        elif mode == 'double':
            neighbors = set(np.argwhere((A.T[:,u] == 1) & (A[:,u] == 1)).T[0])
        else:
            raise ValueError(f'mode {mode} not implemented')
        if exclude is not None:
            neighbors = neighbors.difference(exclude)

        if return_mode == 'list':
            return list(neighbors)
        elif return_mode == 'set':
            return neighbors
        else:
            raise ValueError(f'return_mode {return_mode} not implemented')
    
    if mode == 'single':
        if A_og[a,b] != 1:
            raise ValueError(f'Adjacency matrix error: element [{a}, {b}] not 1')
        if A_og[a,b] == A_og[b,a]:
            raise ValueError(f'Adjacency matrix error: element [{a}, {b}] is a double link, while mode is single')
    elif mode == 'double':
        if A_og[a,b] != 1:
            raise ValueError(f'Adjacency matrix error: element [{a}, {b}] not 1')
        if A_og[a,b] != A_og[b,a]:
            raise ValueError(f'Adjacency matrix error: element [{a}, {b}] is a single link, while mode is double')
    A = A_og.copy()
    # avoid to pick the edge twice
    A[a,b] = -1

    # avoid trivial swaps
    A[a,:] = -1
    A[:,b] = -1
    # remove a and b in and out neighbors to avoid self loops
    A[:, a] = -1
    A[b, :] = -1
    # remove parallel and double edges
    a_neigh = collect_neighbors(A_og, a, exclude = {b}, mode = 'all')
    b_neigh = collect_neighbors(A_og, b, exclude = {a}, mode = 'all')

    a_neigh_in_neigh = collect_neighbors(A_og, a_neigh, exclude = {a,b}, mode = 'in', return_mode = 'list')
    b_neigh_out_neigh = collect_neighbors(A_og, b_neigh, exclude = {a,b}, mode = 'out', return_mode = 'list')

    a_prod = np.transpose([np.tile(a_neigh_in_neigh, len(a_neigh)), np.repeat(a_neigh, len(a_neigh_in_neigh))])
    b_prod = np.transpose([np.tile(b_neigh, len(b_neigh_out_neigh)), np.repeat(b_neigh_out_neigh, len(b_neigh))])
    # print(a_prod)
    # print(b_prod)
    if a_prod.shape[0]: A[tuple(a_prod.T)] = -1
    if b_prod.shape[0]: A[tuple(b_prod.T)] = -1
    if mode == 'single':
        double_edges = np.argwhere((A_og.T == 1) & (A_og == 1))
        A[tuple(double_edges.T)] = -1
    elif mode == 'double':
        single_edges = np.argwhere((A_og.T == 0) & (A_og == 1))
        A[tuple(single_edges.T)] = -1
    else:
        raise ValueError(f'mode {mode} not implemented')
    # print(A)
    return A


def motif_contained_in_adj_fast(adjacency_matrix):
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
  G = ig.Graph.Adjacency(adjacency_matrix.tolist(), mode = 'directed')
  triads = G.triad_census()
  for igraph_motif in igraph_to_nx_translate:
    nx_motif_label = igraph_to_nx_translate[igraph_motif]
    motif_count[nx_motif_label] = triads[igraph_motif]

  return motif_count

def get_triads(G):
    vertices = [v for v in G.vs]
    triads = itertools.combinations(vertices, 3)
    return triads
def get_original_motif_count(experiments):
    motif_name_order = ['S1', 'S2', 'S3', 'S4', 'S5', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']
    original = experiments[experiments['ID'] == 'Original']
    original_triads = original['Triad Census'].iloc[0]
    motif_count_original = np.array([original_triads[motif_name] for motif_name in motif_name_order])
    return np.array(motif_count_original)

def get_random_ensamble_motif_count(experiments):
    motif_name_order = ['S1', 'S2', 'S3', 'S4', 'S5', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']
    random = experiments[experiments['ID'] != 'Original']['Triad Census']
    motifs_random_ensamble = []
    for r in random:
        motifs_random_ensamble.append([r[motif_name] for motif_name in motif_name_order])
    return motifs_random_ensamble

def calculate_z_score(original, random_ensamble):
    # Calculate Z-score
    motif_name_order = ['S1', 'S2', 'S3', 'S4', 'S5', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']
    random_avg = np.mean(random_ensamble, axis = 0)
    random_std = np.std(random_ensamble, axis = 0)

    Z_score = np.divide((original - random_avg), random_std, out=np.zeros(original.shape), where=random_std!=0)
    return Z_score, np.array(motif_name_order)

def calculate_z_score_from_experiments(experiments):
        original_motif_count = get_original_motif_count(experiments)
        random_motif_count = get_random_ensamble_motif_count(experiments)
        z_score, motif_order = calculate_z_score(original_motif_count, random_motif_count)
        return z_score, motif_order

def get_in_degree_and_adjacency_matrix(graph):
    """
    Extracts the adjacency matrix and returns a vector of in-degrees for each node,
    along with the adjacency matrix of the graph.

    Parameters:
    - graph (ig.Graph): An igraph graph object.

    Returns:
    - np.array: A vector of in-degrees for each node.
    - np.array: The adjacency matrix of the graph.
    """
    k_minus_1 = np.array([1 / v.indegree() if v['ECO'] == 1.0 and v.indegree() > 0 else 0 for v in graph.vs])

    # Get the adjacency matrix
    adjacency_matrix = np.array(graph.get_adjacency().data)

    # Get the binarized adjacency matrix
    A = np.where(adjacency_matrix > 0, 1, 0)
    return k_minus_1, A

def get_weighted_in_degree_and_adjacency_matrix(graph):
    """
    Extracts the weighted adjacency matrix and returns a vector of weighted in-degrees for each node,
    along with the binarized adjacency matrix of the graph.

    Parameters:
    - graph (ig.Graph): An igraph graph object.

    Returns:
    - np.array: A vector of weighted in-degrees for each node.
    - np.array: The binarized adjacency matrix of the graph.
    """
    V = graph.vcount()

    # Initialize an empty adjacency matrix
    A = np.zeros((V, V))

    # Fill the weighted adjacency matrix
    for edge in graph.es:
        source, target = edge.source, edge.target
        weight = edge['weight'] if 'weight' in edge.attributes() else (edge['proportion'] if 'proportion' in edge.attributes() else 1)
        A[source, target] = weight
        if not graph.is_directed():
            A[target, source] = weight  # For undirected graphs

    # Calculate weighted in-degree (sum of weights of incoming edges) for each node
    # For directed graphs, sum over rows. For undirected, it's the same as summing over columns.
    k_minus_1 = np.sum(A, axis=0)

    # Apply condition based on 'ECO' attribute
    k_minus_1 = np.array([1 / weight if graph.vs[node]['ECO'] == 1.0 and weight > 0 else 0 for node, weight in enumerate(k_minus_1)])

    return k_minus_1, A

def get_linkage_data(G, linkage_function, matrix_mode = 'both'):

    _, matrix = get_in_degree_and_adjacency_matrix(G)

    if matrix_mode == 'preys':
        I = matrix.copy()
    elif matrix_mode == 'predators':
        I = matrix.T.copy()
    elif matrix_mode == 'both':
        preys = matrix.copy()
        predators = preys.T
        I = np.concatenate((preys, predators), axis = 1)
        del(preys)
        del(predators)
    else:
        raise ValueError(f"matrix_mode {matrix_mode} not implemented.")
    linkage_data = linkage_function(I)
    return linkage_data

def get_communities_from_clusters(clusters, names):
    communities_names = np.unique(clusters)
    communities = {}
    for community in communities_names:
        community_members = np.where(clusters == community)[0]
        community_members = [names[i] for i in community_members]
        communities[community] = community_members
    return communities

def get_combinations(G, n = 3):
    vertices = G.vs
    return list(itertools.combinations(vertices, n))

def get_subgraph_role(G, sub_v, motif, key_v = lambda v: v.index):
    if len(sub_v) != motif.vcount():
        raise ValueError('sub_v and motif must have the same length')
    subgraph = G.subgraph(sub_v)
    # We have to keep track of the original vertex index when we select a key.
    subgraph.vs['key_v'] = [key_v(v) for v in sub_v]
    if subgraph.isomorphic(motif) is False:
        return None
    roles = {v['key_v']: None for v in subgraph.vs}

    for v in subgraph.vs:
        deg_seq = (v.indegree(), v.outdegree())
        for u in motif.vs:
            motif_node_deg_seq = (u.indegree(), u.outdegree())
            if deg_seq == motif_node_deg_seq:
                roles[v['key_v']] = u['role']
    return roles
def prepare_subgraph_role(G, motif_roles, key_v = lambda v: v.index):
    roles_dataframe = {key_v(v): {motif: 0 for sublist in motif_roles for motif in sublist} for v in G.vs}
    return roles_dataframe

def n_reachable_pairs(G):
    # Apply transitive closure
    G_nx = G.to_networkx()
    G_nx = nx.transitive_closure(G_nx)
    # Count edges
    edges_close = G_nx.number_of_edges()
    return edges_close

def f_G(G):
  C = G.components(mode='strong')
  C_card = [len(c_i) for c_i in C]
  return np.sum(np.array([C_i * (C_i - 1) / 2 for C_i in C_card]))

def remove_v_function(G, v, function=n_reachable_pairs):
    """
    Measuring the graph connectivity AFTER the removal of v
    G is the graph
    v is the vertex to be tested
    f_G is the connectivity funciton measured AFTER the removal of from G.
    """
    G_temp = G.copy()
    G_temp.delete_vertices(v['name'])
    return function(G_temp)

def min_index_attr(nodes, f_G_del, attr = lambda v: v['trophic_level'], all=False, reverse = False):
    """
    This function sorts the nodes according to the measure f_G_del made upon the strategy
    and the indices of the vertex, so that if more than one vertex has the same f_G_del, then one with
    the least index is selected.
    """
    experiment = zip(nodes, f_G_del)
    experiment = sorted(experiment, key = lambda x: x[0].index)
    experiment = sorted(experiment, key = lambda x: attr(x[0]))
    experiment = sorted(experiment, key = lambda x: x[1], reverse=reverse)
    if all: 
        return [el[0] for el in experiment] # Return all nodes
    else: 
        return experiment[0][0] # Return the first node of the list

def get_critical_nodes_sequence(G,
                          measure_v = lambda G, v: remove_v_function(G, v, function = n_reachable_pairs),
                          vertices_collecting_function = lambda G: [v for v in G.vs() if v['ECO'] == 1],
                          vertices_ordering_function = lambda nodes, f_G_del: min_index_attr(nodes, f_G_del, reverse = False),
                          preprocessing_G_function = lambda G: G.copy(),
                          stop_condition = lambda G: len([v for v in G.vs() if v['ECO'] == 1])
                          ):
    """
    This function takes a graph and a strategy and returns the list of vertices that minimize the
    connectivity function according to the strategy.
    A strategy is composed by these elements:
    1. a measure on the vertices;
    2. a function that collects the vertices from the graph given the properties of the vertex;
    3. a function that sorts and collects the vertices given the measure;

    With a combination of these 4 elements it is possible to test different strategies.
    For example, one may test how does it change the connectivity value of the graph
    when the vertices that minimize the most the connectivity value are removed first.
    In this case, the measure on the vertices is the connectivity function given the graph after the removal of a vertex;
    the function that collects the vertices is a function that takes all vertices of the graph regardless their properties;
    the sorting and collecting function sorts the vertices according to the measured connectivity function after their removal and collect
    the vertex that minimize the most the connectivity; the loop stop when there are no more vertices in the graph, i.e. when the graph is empty.
    """

    # 1 Initialization
    # 1.1 List to be returned: the nodes given the ordering and the function
    list_of_removed_nodes = []
    # 1.2 G.delete_vertices in an inplace function, therefore a dummy variable is needed
    G_M = G.copy()
    
    # 2. Loop on the graph given a condition
    while(stop_condition(G_M)):
    # 2.1 Collect the nodes with a fixed order
        v_order = vertices_collecting_function(G_M)

    # 2.2 Measure the vertices property given by measure_v
        G_prep = preprocessing_G_function(G_M)
        f_G_del = [measure_v(G_prep, v) for v in v_order]
    # 3.3 Sort and select the vertices
        selected_v_s = vertices_ordering_function(v_order, f_G_del)
    # 3.4 Handling different strategies: vertices_ordering_function can return a single vertex of a list of vertices
        if type(selected_v_s) == list:
            list_of_removed_nodes = [v['name'] for v in selected_v_s]
        else:
            list_of_removed_nodes.append(selected_v_s['name'])
    # Node(s) is removed to meet the stop condition
        G_M.delete_vertices(selected_v_s)
    
    return list_of_removed_nodes


def robustness_function_reachable_pairs(df_node_removal_G):
  values = df_node_removal_G['Fraction of reachable pairs'].to_numpy()
  N = df_node_removal_G.shape[0]
  return np.sum(values) / N

def rho_G_values_for_each_node(G, node_sequence):
    # Get all nodes of the food web
    S = G.vcount()
    # Define the function
    r_G = lambda G_i: n_reachable_pairs(G_i) / (S * S)
    # Initialize the function
    r_G_node_sequence = [r_G(G)]
    G_temp = G.copy()
    for v in node_sequence:
        G_temp.delete_vertices(v)
        r_G_node_sequence.append(r_G(G_temp))
    return r_G_node_sequence


def rho_G(G, node_sequence):
    # Get all nodes of the food web
    S = G.vcount()
    # Define the function
    r_G = lambda G_i: n_reachable_pairs(G_i) / (S * S)
    # Initialize the sum
    sum_r_G = r_G(G)
    G_temp = G.copy()
    for v in node_sequence:
        G_temp.delete_vertices(v)
        sum_r_G += r_G(G_temp)
    # Normalize by the number of living species
    S_G = len([v for v in G.vs if v['ECO'] == 1])
    sum_r_G /= S_G
    return sum_r_G

def find_food_chain(G):
    # Get rid of non-living organic matter nodes
    G.delete_vertices([v.index for v in G.vs if v['ECO'] == 2])

    # Get rid of self loops
    G.delete_edges([e.index for e in G.es if e.source == e.target])

    # Find all the food chains
    food_chains = []

    # Find basal species
    basal_species = [v.index for v in G.vs if v.indegree() == 0]
    path_function = lambda G, basal, v: G.get_all_shortest_paths(basal, to=v.index)
    select_path_function = lambda path: len(set(path)) == len(path)

    for basal in basal_species:
        valid_paths = [path for v in G.vs for path in path_function(G, basal, v) if select_path_function(path) if v.index not in basal_species]
        food_chains += valid_paths

    return food_chains