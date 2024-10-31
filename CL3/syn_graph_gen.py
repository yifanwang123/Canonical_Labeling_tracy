import numpy as np
import networkx as nx
import random
import os
from networkx.algorithms import isomorphism
import multiprocessing
# num_nodes = 9

# edge_list = [
#     [0, 1], [0, 3],
#     [1, 2], [1, 0],
#     [2, 1], [2, 5],
#     [3, 0], [3, 1], [3, 4], [3, 6], [3, 7],
#     [4, 1], [4, 3], [4, 5], [4, 7],
#     [5, 1], [5, 2], [5, 4], [5, 8], [5, 7],
#     [6, 3], [6, 7],
#     [7, 8], [7, 7],
#     [8, 5], [8, 7]
# ]

# G = nx.Graph()

# # Add nodes to the graph
# G.add_nodes_from([i for i in range(num_nodes)])

# # Add edges to the graph
# G.add_edges_from(edge_list)

def shuffle_sequence(seq):
    shuffled_seq = seq[:]
    random.shuffle(shuffled_seq)
    return shuffled_seq



def generate_graph_with_custom_degree_distribution(n, degree_choices):
    """
    Generates a graph with n nodes where each node's degree (number of neighbors)
    is randomly chosen from a predefined list of possible degrees.
    :param n: Number of nodes.
    :param degree_choices: List of possible degrees (neighbor sizes).
    :return: A graph with nodes having random degrees from degree_choices.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(n))  # Add n nodes
    
    # Assign each node a random degree from the predefined list
    degrees = {node: random.choice(degree_choices) for node in G.nodes()}
    
    # Create edges while respecting the degree assigned to each node
    for node in G.nodes():
        neighbors_needed = degrees[node]  # The number of neighbors this node should have
        possible_neighbors = [n for n in G.nodes() if n != node and G.degree(n) < degrees[n]]
        
        # Randomly sample neighbors and create edges
        if len(possible_neighbors) >= neighbors_needed:
            neighbors = random.sample(possible_neighbors, neighbors_needed)
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
    
    return G


def generate_non_isomorphic_er_graphs_with_same_edges(n, p):
    # """
    # Generates two non-isomorphic Erdős-Rényi (ER) graphs with the same number of edges.
    # :param n: Number of nodes in both graphs.
    # :param p: Probability for edge creation in the ER model.
    # :return: Two non-isomorphic graphs with the same number of edges.
    # """
    # while True:
    #     # Generate two random Erdős-Rényi graphs with the same number of nodes
    #     graph_1 = nx.erdos_renyi_graph(n, p, seed=random.randint(0, 1000))
    #     graph_2 = nx.erdos_renyi_graph(n, p, seed=random.randint(1001, 2000))

    #     # Adjust edges to make them have the same number of edges
    #     num_edges_graph_1 = graph_1.number_of_edges()
    #     num_edges_graph_2 = graph_2.number_of_edges()

    #     # If graph_1 has more edges than graph_2, remove some from graph_1
    #     if num_edges_graph_1 > num_edges_graph_2:
    #         edges_to_remove = random.sample(list(graph_1.edges), num_edges_graph_1 - num_edges_graph_2)
    #         graph_1.remove_edges_from(edges_to_remove)
    #     # If graph_2 has more edges than graph_1, remove some from graph_2
    #     elif num_edges_graph_2 > num_edges_graph_1:
    #         edges_to_remove = random.sample(list(graph_2.edges), num_edges_graph_2 - num_edges_graph_1)
    #         graph_2.remove_edges_from(edges_to_remove)

    #     # Check if they are isomorphic
    #     GM = isomorphism.GraphMatcher(graph_1, graph_2)
    #     if not GM.is_isomorphic():
    #         return graph_1, graph_2
    degree_max = int(n*0.2)
    degree_choices = [random.randint(5, 10) for i in range(degree_max)]

    while True:
        # Generate two random graphs with the same number of nodes and custom degree distributions
        graph_1 = generate_graph_with_custom_degree_distribution(n, degree_choices)
        graph_2 = generate_graph_with_custom_degree_distribution(n, degree_choices)

        # Check if they are isomorphic
        GM = nx.isomorphism.GraphMatcher(graph_1, graph_2)
        if not GM.is_isomorphic():
            return graph_1, graph_2

def generate_multiple_non_isomorphic_graph_pairs(sample_id, n, p, dir):

    if not os.path.exists(dir):
        os.makedirs(dir)
    graph_1, graph_2 = generate_non_isomorphic_er_graphs_with_same_edges(n, p)
    
    adjacency_matrix_1 = nx.adjacency_matrix(graph_1).todense()
    adjacency_matrix_2 = nx.adjacency_matrix(graph_2).todense()
    print('finish')
    np.save(f'{dir}/sample{sample_id}-non-1', adjacency_matrix_1)
    np.save(f'{dir}/sample{sample_id}-non-2', adjacency_matrix_2)


# path_dir = '/home/cds/Yifan/canonical_labeling/data/self_generated_data/n100_er2_non'
# n = 100
# num_pair = [i for i in range(200)] 
# p = 0.2

# # generate_multiple_non_isomorphic_graph_pairs(num_pair, n, p, path_dir)

# def worker(sample_id):
#     a = generate_multiple_non_isomorphic_graph_pairs(sample_id, n, p, path_dir)


# with multiprocessing.Pool(processes=40) as pool:
#     pool.map(worker, num_pair, chunksize=1)





# def iso(sample_id, n, p, dir):

#     if not os.path.exists(dir):
#         os.makedirs(dir)
#     # g_1 = nx.erdos_renyi_graph(n, p)
#     degree_choices = [random.randint(5, 10) for i in range(20)]

#     g_1 = generate_graph_with_custom_degree_distribution(n, degree_choices)

#     adjacency_matrix_1 = nx.adjacency_matrix(g_1).todense()
#     new_order = shuffle_sequence([i for i in range(n)])
#     old_labels = list(g_1.nodes())
#     mapping = {old_labels[i]: new_order[i] for i in range(len(old_labels))}
#     # print(mapping)
#     g_2 = nx.relabel_nodes(g_1, mapping)
#     adj_matrix_2 = nx.adjacency_matrix(g_2, nodelist=sorted(g_2.nodes())).todense()
#     is_isomorphic = nx.is_isomorphic(g_1, g_2)
#     if is_isomorphic:
#         np.save(f'{dir}/sample{sample_id}-1', adjacency_matrix_1)
#         np.save(f'{dir}/sample{sample_id}-2', adj_matrix_2)

def iso(sample_id, n, p, dir_path):

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Step 1: Generate a random directed graph G
    G = nx.gnp_random_graph(n, p, directed=True)

    # Step 2: Generate a random permutation of the nodes
    nodes = list(G.nodes())
    permuted_nodes = nodes.copy()
    random.shuffle(permuted_nodes)
    mapping = dict(zip(nodes, permuted_nodes))

    # Step 3: Relabel nodes according to the permutation
    G_prime = nx.relabel_nodes(G, mapping)

    # Step 4: Ensure consistent node ordering when extracting adjacency matrices
    # For G
    adjacency_matrix_G = nx.adjacency_matrix(G, nodelist=sorted(nodes)).todense()
    # For G_prime
    adjacency_matrix_G_prime = nx.adjacency_matrix(G_prime, nodelist=sorted(permuted_nodes)).todense()

    # Step 5: Check isomorphism
    is_isomorphic = nx.is_isomorphic(G, G_prime)

    if is_isomorphic:
        np.save(f'{dir_path}/sample{sample_id}-1', adjacency_matrix_G)
        np.save(f'{dir_path}/sample{sample_id}-2', adjacency_matrix_G_prime)
    else:
        print(f"Graphs are not isomorphic for sample {sample_id}")

n = 10

num_pair_1 = 30
num_pair = [i for i in range(num_pair_1)]
path_dir = f'/home/cds/Documents/canonical_labeling/data/self_generated_data/n{n}_{num_pair_1}_predefined'
p = 0.2

generate_multiple_non_isomorphic_graph_pairs(num_pair, n, p, path_dir)

def worker(sample_id):
    try:
        a = generate_multiple_non_isomorphic_graph_pairs(sample_id, n, p, path_dir)
    except Exception as e:
        print(f"Error in worker with sample_id {sample_id}: {e}")

with multiprocessing.Pool(processes=10) as pool:
    pool.map(worker, num_pair, chunksize=1)
    pool.close()
    pool.join()
print('finish non-iso pairs')


def worker(sample_id):
    a = iso(sample_id, n, p, path_dir)


with multiprocessing.Pool(processes=10) as pool:
    pool.map(worker, num_pair, chunksize=1)
    pool.close()
    pool.join()