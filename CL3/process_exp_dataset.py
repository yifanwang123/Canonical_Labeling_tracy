
import os
import numpy as np
import sys
# directory = '/home/cds/Yifan/canonical_labeling/data/EXP_dataset/raw'

# exp_dataset = '/home/cds/Yifan/canonical_labeling/data/EXP_dataset/raw/EXP'
# cexp_dataset = '/home/cds/Yifan/canonical_labeling/data/EXP_dataset/raw/CEXP'


def read_graphs_from_file(filename):
    graphs = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    idx = 0
    while idx < len(lines):
        # Read the number of nodes and label of the graph
        n, l = map(int, lines[idx].strip().split())
        idx += 1
        
        # Initialize adjacency matrix
        adj_matrix = np.zeros((n, n), dtype=int)
        
        # Read node information
        for i in range(n):
            node_info = list(map(int, lines[idx].strip().split()))
            t, m = node_info[0], node_info[1]
            neighbors = node_info[2:]
            # print(neighbors)
            
            # Fill adjacency matrix
            for neighbor in neighbors:
                adj_matrix[i][neighbor] = 1
            
            idx += 1
        
        
        
        # Store the adjacency matrix and graph label

        graphs.append((adj_matrix, l))
        # if len(graphs) == 2:
        #     sys.exit()
    
    return graphs

def save_graphs(graphs, output_dir, output_prefix="sample"):
    sample_id = 0
    for i, (adj_matrix, label) in enumerate(graphs):
        if i %2 == 0:
            sample_id += 1

        filename = os.path.join(output_dir, f'{output_prefix}{sample_id}-{label}.npy')
        np.save(filename, adj_matrix)
        print(f"Graph {i} with label {label} saved to {filename}")


data = 'EXP'
raw_path = os.path.join('/home/cds/Yifan/canonical_labeling/data/EXP_dataset/raw', data, 'GRAPHSAT.txt')
output_dir = f'/home/cds/Yifan/canonical_labeling/data/EXP_dataset/{data}'
graphs = read_graphs_from_file(raw_path)
save_graphs(graphs, output_dir)

