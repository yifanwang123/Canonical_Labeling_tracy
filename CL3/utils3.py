import networkx as nx
import numpy as np
import random
import sys
import matplotlib.pyplot as plt

class Candidate:
    def __init__(self, lab, invlab, singcode, code, sortedlab=False, next=None):
        self.lab = lab
        self.invlab = invlab
        self.singcode = singcode
        self.code = code
        self.sortedlab = sortedlab
        self.next = next


class TracesVars:
    def __init__(self, graph, options=None, stats=None):
        self.graph = graph
        self.options = options
        self.stats = stats
        self.partition = None
        self.node_invariant = None
        self.candidates = None
        self.seen_codes = set()
        self.generators = []
        self.n = graph.num_nodes
        self.mark = 0
        self.stackmark = 0
        self.maxdeg = 0
        self.mindeg = self.n



class NodeInvariant:
    def __init__(self, graph):
        self.graph = graph
    
    def compute_invariant(self, cls):
        invariant = []
        cell_structure = {}
        for idx, cell_id in enumerate(cls):
            if cell_id not in cell_structure:
                cell_structure[cell_id] = []
            cell_structure[cell_id].append(idx)
        for cell_id in sorted(cell_structure):
            cell = cell_structure[cell_id]
            # sys.exit()
            cell_invariant = self.cell_invariant(cell, cls)
            invariant.append(cell_invariant)
        # return tuple(sorted(invariant))
        return tuple(sorted(invariant))


    def cell_invariant(self, cell, cls):
        # cell = 
        cell_neighbors = []
        for vertex in cell:
            neighbors = self.graph.neighbor_dict[vertex]


            neighbor_degrees = sorted([[len(self.graph.neighbor_dict[neighbor]), cls[neighbor]] for neighbor in neighbors])
            # print(vertex, neighbor_degrees)
            cell_neighbors.append(tuple(neighbor_degrees))
        return tuple(sorted(cell_neighbors))


    # def cell_invariant(self, cell, cls):
    #     # cell = 
    #     cell_neighbors = []
    #     for vertex in cell:
    #         neighbors = self.graph.neighbor_dict[vertex]


    #         neighbor_degrees = sorted(tuple(tuple((len(self.graph.neighbor_dict[neighbor]), cls[neighbor]))) for neighbor in neighbors)
    #         # print(vertex, neighbor_degrees)
    #         cell_neighbors.append(tuple(neighbor_degrees))
    #     return tuple(sorted(cell_neighbors))
    
    def helper(self, cell, cls):
        cell_neighbors = {}
        for vertex in cell:
            neighbors = self.graph.neighbor_dict[vertex]
            # lst = [[len(self.graph.neighbor_dict[neighbor], ] for neighbor in neighbors]
            # cls_dict = {}
            # for neighbor in neighbors:
            #     if cls[neighbor] not in cls_dict:
            #         cls_dict[cls[neighbor]] = 1
            #     else:
            #         cls_dict[cls[neighbor]] += 1
            # cls_dict = dict(sorted(cls_dict.items(), key=lambda x: x[1]))
            # # values = list(cls_dict.values())

            neighbor_degrees = sorted([[len(self.graph.neighbor_dict[neighbor]), cls[neighbor]] for neighbor in neighbors])
            cell_neighbors[vertex] = neighbor_degrees
        cell = sorted(cell, key=lambda x: cell_neighbors[x])
        return cell
    
    def node_each_cell_invariant(self, cell, cls):

        new_neighbor_rank = {}

        for vertex in cell:
            # new_neighbor_rank[vertex] = self.helper1(self.graph.neighbor_dict[vertex], cls)
            new_neighbor_rank[vertex] = self.helper(self.graph.neighbor_dict[vertex], cls)

        # print(new_neighbor_rank)
        cell_neighbors = {}
        another = {}
        for vertex in cell:
            neighbor_num_neighbor = [self.helper3(self.graph.neighbor_dict[neighbor]) for neighbor in new_neighbor_rank[vertex]]
            another[vertex] = neighbor_num_neighbor
            neighbor_degrees = sorted([[len(self.graph.neighbor_dict[neighbor]), cls[neighbor]] for neighbor in new_neighbor_rank[vertex]])
            cell_neighbors[vertex] = neighbor_degrees
        cell = sorted(cell, key=lambda x: (cell_neighbors[x], another[x]), reverse=True)
        return cell

    def helper3(self, neighbors):
        lst = []
        for node in neighbors:
            lst.append(len(self.graph.neighbor_dict[node]))

        return sorted(lst)
    
    def quotient_graph(self, original_graph, partition):
        G = nx.Graph()
        for cell_id, nodes in partition.cell_dict.items():
            nodes = self.node_each_cell_invariant(nodes, partition.cls)
            size = len(nodes)
            # in_neighbors = 0
            # out_neighbors = 0
            in_neighbors_list = []
            neighbor_num_list = []
            for node in nodes:
                neighbors_nodes = list(original_graph.graph.neighbors(node))
                # print(neighbors_nodes)
                in_neighbors = sorted(tuple(len(original_graph.neighbor_dict[i]) for i in neighbors_nodes))
                # print(in_neighbors)
                in_neighbors_list.append(tuple(in_neighbors))


                neighbors_num = len(list(original_graph.graph.neighbors(node)))
                neighbor_num_list.append(neighbors_num)
                # out_neighbors_nodes = list(original_graph.graph.successors(node))
                # out_neighbors = sorted(list(len(original_graph.neighbor_dict[i]) for i in out_neighbors_nodes))
            # in_neighbors_list = sorted(in_neighbors_list)

            # G.add_node(cell_id, label=f'cell_{cell_id}', size=size, nei=tuple(in_neighbors_list))
            G.add_node(cell_id, label=f'cell_{cell_id}', size=size, nei=tuple(sorted(neighbor_num_list)))

        
        # Initialize edge labels (number of edges between cells)
        edge_labels = {}
        # For each edge in G, find the cells of its endpoints
        for u, v in original_graph.graph.edges():
            cell_u = partition.cls[u]
            cell_v = partition.cls[v]
            # Edge between cells cell_u and cell_v

            key = (cell_u, cell_v)
            if key not in edge_labels:
                edge_labels[key] = 0
            edge_labels[key] += 1

        # Add edges to Q with labels
        for (cell_u, cell_v), weight in edge_labels.items():
            G.add_edge(cell_u, cell_v, weight=weight)
        return G

    def quotien_graph_info(self, original_graph, partition, sequence):

        quotient_graph = self.quotient_graph(original_graph, partition)
        
        # # Extract cell sizes
        # cells = []
        # for node, data in quotient_graph.nodes(data=True):
        #     cell_size = data.get('size', 1)
        #     cells.append((node, cell_size))
        # # Sort cells by node index to ensure consistent ordering
        # cells.sort()

        # edges = []
        # for u, v, data in quotient_graph.edges(data=True):
        #     weight = data.get('weight', 1)
        #     # For undirected graphs, ensure consistent ordering of nodes
        #     edge = tuple(sorted((u, v)) + (weight,))
        #     edges.append(edge)
        # # Sort edges to ensure consistent ordering
        # edges.sort()

        # # Create the representation as a tuple
        # representation = (tuple(cells), tuple(edges))

        # Extract cell sizes
        cells = [(node, data.get('size'), data.get('nei'), tuple(sorted(list(quotient_graph.neighbors(node))))) for node, data in quotient_graph.nodes(data=True)]
        cells.sort()  # Sort by node index for consistency
        # print(cells)

        # Extract edges with weights
        edges = []
        for u, v, data in quotient_graph.edges(data=True):
            weight = data.get('weight')
            # edge = (min(u, v), max(u, v), weight)  # Sort node indices in edge
            edge = (u, v, weight)
            edges.append(edge)
        edges.sort()  # Sort edges for consistency

        # Build canonical representation
        # cell_str = ';'.join(f'{node}:{size}' for node, size in cells)
        # edge_str = ';'.join(f'{u}-{v}:{weight}' for u, v, weight in edges)
        # representation = f'[{cell_str}][{edge_str}]'
        cell_info = tuple(sorted((node, size, in_nei, nei_size) for node, size, in_nei, nei_size in cells))
        edge_info = tuple(sorted((u, v, weight) for u, v, weight in edges))
        cls_info = self.compute_invariant(partition.cls)
        indi_node_cls = partition.cls[sequence[0][0]]
        representation = tuple((indi_node_cls, cell_info, edge_info))
        # representation = cls_info
        # print(type(representation))
        # sys.exit()
        return representation

    






class Partition:
    def __init__(self, cls, inv, cell_dict, active, cells, code):
        '''
        cls: List of class sizes. cls[i] indicates the node i's class id. 
        ind: List of position and coloring mapping.
        cell_dict: cell_dict[i] = [node1, ..., node_k]
        active: Indicates if the partition is active.
        cells: Number of cells in the partition.
        code: A code representing the partition state (node invariant function).
        '''
        self.cls = cls  
        self.inv = inv
        self.cell_dict = cell_dict
        self.active = active
        self.cells = cells
        self.code = code

def initialize_part(graph):
    '''
    Random generated
    '''
    n = graph.num_nodes
    nodes = [i for i in range(n)]
    num_cells = 1
    cell_dict = {}
    # random.shuffle(nodes)
    
    
    # for i in range(num_cells):
    #     cell_dict[i] = [nodes.pop()]

    # for element in nodes:
    #     random.choice(list(cell_dict.values())).append(element)    
    
    cell_dict[0] = nodes
    inv = [float('-inf')] * n
    cls_ = [0] * n

    for cell_id in cell_dict.keys():
        for node in cell_dict[cell_id]:
            cls_[node] = cell_id
    

    lst = [0] * num_cells
    for cell in cell_dict.keys():
        lst[cell] = len(cell_dict[cell])

    for i in range(n):
        cell_id = cls_[i]
        inv[i] = 1 + sum(lst[:cell_id])

    partition = Partition(cls=cls_, inv=inv, cell_dict=cell_dict, active=1, cells=num_cells, code=None)
    # print(cell_dict)
    # sys.exit()
    return partition


class LabeledGraph:
    def __init__(self, matrix_file, dataset=None):
        '''
        node index start from 0
        '''
        if dataset == 'self_generated_data' or dataset == 'EXP_dataset':
            self.graph, self.num_nodes = self.read_from_file(matrix_file)
        else:
            if dataset == 'benchmark1_false' or dataset == 'benchmark1_true':
                self.graph, self.num_nodes = self.read_from_file1(matrix_file)
            else:
                self.graph, self.num_nodes = self.read_from_file2(matrix_file)

        self.neighbor_dict = {node: list(self.graph.neighbors(node)) for node in self.graph.nodes()}
        # print(self.neighbor_dict)
        # adj_matrix = nx.adjacency_matrix(self.graph)

        # # Convert to a dense format for printing
        # adj_matrix_dense = adj_matrix.todense()
        # print(adj_matrix_dense)
        # for i, row in enumerate(adj_matrix_dense):
        #     if np.sum(row) == 4:
        #         print(f"Node {i}: {row}")
        # print(self.num_nodes)
        # print(self.neighbor_dict)
        # sys.exit()
        # self.colors = range(self.num_nodes)
        
    def read_from_file(self, file_path):
        matrix = np.load(file_path)
        nodes = range(len(matrix))
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if val != 0:
                    graph.add_edge(i, j)
        return graph, len(nodes)

    def read_from_file1(self, file_path):
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
        return G, G.number_of_nodes()
    
    def read_from_file2(self, file_path):
        # G = nx.DiGraph()

        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        first_line = lines[0]
        n = int(first_line.split()[1].split('=')[1])
        
        G = nx.Graph()
        
        # Process each line to add edges
        for line in lines[1:]:
            if ':' in line:
                parts = line.split(':')
                node = int(parts[0]) - 1
                neighbors = parts[1].strip().replace('.', '').split()
                neighbors = list(map(int, neighbors))
                for neighbor in neighbors:
                    G.add_edge(node, neighbor-1)
        
        return G, n



    def get_node_labels(self):
        return nx.get_node_attributes(self.graph, 'label')


    
    def can_graph(self, graph):
        self.graph = graph
        self.neighbor_dict = {node: list(self.graph.neighbors(node)) for node in self.graph.nodes()}
        # print(self.neighbor_dict)



class TreeNode:
    def __init__(self, partition, sequence, parent=None, invariant=None):
        self.partition = partition 
        self.parent = parent  
        self.children = []
        self.sequence = sequence
        self.invariant = invariant
        self.prune = 0

    def add_child(self, child_node):
        self.children.append(child_node)


def add_edges(graph, node):
    for child in node.children:
        graph.add_edge(node, child)
        add_edges(graph, child)

# def cls_list(node):
    
#     for child in node.children:



def get_leaf_invariants(node):
    leaf_invariants = []
    if not node.children:  # Leaf node
        if node.prune == 0:
            leaf_invariants.append((str(node.sequence), tuple(node.invariant)))
    else:
        for child in node.children:
            leaf_invariants.extend(get_leaf_invariants(child))
    return leaf_invariants


def draw_search_tree(root, idx):
    G = nx.DiGraph()
    add_edges(G, root)

    # Position nodes using spring layout
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    # Draw the graph
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=False, node_size=20, node_color="lightblue", font_size=5, font_weight="bold", arrows=False)

    # Display partition around each node
    # for node in G.nodes():
    #     x, y = pos[node]
    #     plt.text(x, y + 0.1, f"{node.partition.cls}", ha='center', fontsize=8, color="darkblue")

    leaf_invariants = tuple(get_leaf_invariants(root))
    # print(len(leaf_invariants))
    # sys.exit()

    # Sort leaves by invariant lexicographically
    sorted_leaves = sorted(leaf_invariants, key=lambda x: x[1])
    leaf_order = {sequence: order for order, (sequence, _) in enumerate(sorted_leaves, start=1)}

    
    idx_ = 0
    for node in G.nodes():
        idx_ += 1

        x, y = pos[node]
        if str(node.sequence) in leaf_order:
            # Display order number for leaf nodes based on lexicographical order
            plt.text(x, y - 5, f"{leaf_order[str(node.sequence)]}", ha='center', fontsize=5, color="darkred")
        plt.text(x, y + 5, f"{node.sequence}", ha='center', fontsize=5, color="darkblue")
        plt.text(x, y + 5 + idx_, f"{node.partition.cls}", ha='center', fontsize=4, color="darkblue")
        if node.prune != 0:
            plt.text(x, y - 8, f"prune: {node.prune}", ha='center', fontsize=5, color="darkred")


    plt.title("Tree Visualization with Labels and Partition")
    plt.savefig(f"/home/cds2/Yifan/canonical_labeling/CL3/tree_graph_{idx}.pdf", format="pdf")




    # Display the order of leaves based on lexicographical order of their invariants
    # for i, (sequence, invariant) in enumerate(sorted_leaves, start=1):
    #     print(f"Order {i}: Node {sequence}")
    # plt.show()

