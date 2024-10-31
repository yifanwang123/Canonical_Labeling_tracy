import pynauty as nauty
import networkx as nx
import numpy as np


def read_from_file1(file_path):
    G = nx.Graph()

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if parts[0] == 'p':
                continue
            elif parts[0] == 'e':
                nodev = int(parts[1])-1
                nodeu = int(parts[2]) -1
                G.add_edge(nodev, nodeu)
    

    neighbor_dict = {node: list(G.neighbors(node)) for node in G.nodes()}

    return G.number_of_nodes(), neighbor_dict, G



def read_from_file(file_path):
    matrix = np.load(file_path)
    nodes = range(len(matrix))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val != 0:
                graph.add_edge(i, j)

    neighbor_dict = {node: list(graph.neighbors(node)) for node in graph.nodes()}
    
    return len(nodes), neighbor_dict, graph









a, b, g1 = read_from_file('/home/cds/Documents/AAAI25/canonical_labeling/syn_data_2/1/1.npy')
c, d, g2 = read_from_file('/home/cds/Documents/AAAI25/canonical_labeling/syn_data_2/1/2.npy')

graph1 = nauty.Graph(number_of_vertices=a, adjacency_dict=b)
graph2 = nauty.Graph(number_of_vertices=c, adjacency_dict=d)

can_lab1 = nauty.canon_label(graph1)
can_lab2 = nauty.canon_label(graph2)



def check_iso(form1, form2, graph1, graph2):
    for u, v in graph1.edges():
        node1 = form2.index(form1[u])
        node2 = form2.index(form1[v])
        if (node1, node2) in graph2.edges():
            continue
        else:
            return False

    return True


# print(check_iso(can_lab1, can_lab2, g1, g2))
print(can_lab1)
print(can_lab2)

