from utils import Candidate, Partition, TreeNode
from SS_al import initialize_schreier_sims_with_partitions
from sympy.combinatorics import Permutation
import sys
from collections import defaultdict
import copy
import itertools
import networkx as nx
import matplotlib.pyplot as plt


def individualize(partition, vertex):
    n = len(partition.cls)

    new_cls = copy.deepcopy(partition.cls)
    new_inv = copy.deepcopy(partition.inv)
    vertex_class_current = partition.cls[vertex]
    # print(f'individulized node cell id: {vertex_class_current}')

    
    for node in range(n):
        if partition.cls[node] < vertex_class_current or node == vertex:
            new_cls[node] = partition.cls[node]
        else:
            new_cls[node] = partition.cls[node] + 1
        if node in partition.cell_dict[vertex_class_current] and node != vertex:
            new_inv[node] += 1

    cells = partition.cells + 1
    new_dict = {}
    
    for i in range(n):
        if new_cls[i] not in new_dict:
            new_dict[new_cls[i]] = [i]
        else:
            new_dict[new_cls[i]].append(i)

    new_partition = Partition(cls=new_cls, inv=new_inv, cell_dict=new_dict, active=0, cells=cells, code=partition.code)
    # print(new_partition.cell_dict)
    return new_partition


# def one_dimensional_refinement(graph, partition):
#     n = graph.num_nodes
#     cls_ = partition.cls
#     inv = partition.inv

#     refined = True
#     while refined:
#         refined = False
#         new_cls = [0] * n
#         new_inv = [0] * n
#         neighbor_classes = {}

#         for i in range(n):
#             neighbors = tuple(sorted(cls_[neighbor] for neighbor in graph.neighbor_dict[i]))
#             if neighbors not in neighbor_classes:
#                 neighbor_classes[neighbors] = len(neighbor_classes)
#             new_cls[i] = neighbor_classes[neighbors]
#             new_inv[new_cls[i]] = i

#         if new_cls != cls_:
#             refined = True
#             cls_ = new_cls
#             inv = new_inv

#     partition.cls = cls_
#     partition.inv = inv
#     return partition

# def two_dimensional_refinement(graph, partition):
#     n = graph.num_nodes
#     cls = partition.cls
#     inv = partition.inv

#     refined = True
#     while refined:
#         refined = False
#         new_cls = [0] * n
#         new_inv = [0] * n
#         neighbor_pairs = {}

#         for i in range(n):
#             for j in graph.neighbor_dict[i]:
#                 pair = (cls[i], cls[j])
#                 if pair not in neighbor_pairs:
#                     neighbor_pairs[pair] = len(neighbor_pairs)
#                 new_cls[i] = neighbor_pairs[pair]

#         unique_classes = len(neighbor_pairs)
#         if unique_classes > len(new_inv):
#             new_inv = [0] * unique_classes

#         for i in range(n):
#             new_inv[new_cls[i]] = i

#         if new_cls != cls:
#             refined = True
#             cls = new_cls
#             inv = new_inv

#     partition.cls = cls
#     partition.inv = inv
#     return partition



def node_neighbors_color(node_neighbors, cls):
    color_assign = tuple(sorted(cls[neighbor] for neighbor in node_neighbors))
    # print('here!!:', color_assign)
    return color_assign



def find_outer_index(list_of_lists, element):
    for outer_index, inner_list in enumerate(list_of_lists):
        if element in inner_list:
            return outer_index
    return -1



def reorder_graph_based_on_neighbor_degrees(G):
    neighbor_degrees = {node: sum(G.degree(neighbor) for neighbor in G.neighbors(node)) for node in G.nodes()}
    
    sorted_nodes = sorted(G.nodes(), key=lambda x: neighbor_degrees[x], reverse=True)
    
    H = nx.Graph()
    mapping = {node: idx for idx, node in enumerate(sorted_nodes)}
    H.add_nodes_from(range(len(sorted_nodes)))
    H.add_edges_from((mapping[u], mapping[v]) for u, v in G.edges())
    
    return H


def one_dimensional_refinement(G, partition, node_invariant):
    
    n = G.num_nodes
    # print(n)
    curr_cls = copy.deepcopy(partition.cls)

    # a = {}
    # for i in range(n):
    #     a[i] = curr_cls[i]
    # draw_graph(G.graph, a)
    

    # sorted_dict = {k: v for k, v in sorted(partition.cell_dict.items(), key=lambda item: len(item[1]))}
    sorted_dict = {k: v for k, v in sorted(partition.cell_dict.items(), key=lambda item: item[0])}
    # print(f'sorted_dict: {sorted_dict}')
    # sys.exit()
    for cell_id, cell in sorted_dict.items():
        
        if len(cell) == 1:
            continue
        
        cell = node_invariant.node_each_cell_invariant(cell, curr_cls)
        
        # print('here!!:', cell)

        neighbors_colors = {node: node_neighbors_color(G.neighbor_dict[node], curr_cls) for node in cell}
        # print(f'neighbors_colors of cell {cell_id}: {neighbors_colors}')

        new_color_dict = defaultdict(list)
        # print(new_color_dict)
        for node, color in neighbors_colors.items():
            new_color_dict[color].append(node)
        
        new_color_pattern = list(new_color_dict.values())
        # print(new_color_pattern)
        # sys.exit()
        # print('end')
        if len(new_color_pattern) > 1:
            new_cls = copy.deepcopy(curr_cls)

            # print(f'here 1: {curr_cls}', new_color_pattern)
            # for node in range(n):
            #     node_curr_cell_id = curr_cls[node]
            #     if node in cell:
            #         # print(f'here2: node {node}, cell {cell}')
            #         idx = find_outer_index(new_color_pattern, node)
            #         new_cls[node] = node_curr_cell_id + idx
            #     else:
            #         # print('here3:', node, node_curr_cell_id, curr_cls[new_color_pattern[0][0]])
            #         if node_curr_cell_id > curr_cls[new_color_pattern[0][0]]:
            #             new_cls[node] = node_curr_cell_id + len(new_color_pattern) - 1
            # curr_cls = new_cls
        # print(f'here4: {curr_cls}')
            # cell = node_invariant.node_each_cell_invariant(cell)
            
            for node in cell:
                node_curr_cell_id = curr_cls[node]
                idx = find_outer_index(new_color_pattern, node)
                new_cls[node] = node_curr_cell_id + idx
            for node in range(n):
                if node not in cell:
                    node_curr_cell_id = curr_cls[node]
                        # print('here3:', node, node_curr_cell_id, curr_cls[new_color_pattern[0][0]])
                    if node_curr_cell_id > curr_cls[new_color_pattern[0][0]]:
                        new_cls[node] = node_curr_cell_id + len(new_color_pattern) - 1
            curr_cls = new_cls



    # print('finish for loop')
    new_cells = len(set(curr_cls))
    new_inv = [float('-inf')] * n
    
    new_dict = {}

    for node in range(n):
        if curr_cls[node] not in new_dict:
            new_dict[curr_cls[node]] = [node]
        else:
            new_dict[curr_cls[node]].append(node)

    lst = [0] * new_cells


    for cell in new_dict.keys():
        lst[cell] = len(new_dict[cell])

    for i in range(n):
        cell_id = curr_cls[i]
        new_inv[i] = 1 + sum(lst[:cell_id])

    code = node_invariant.compute_invariant(curr_cls)

    new_partition = Partition(cls=curr_cls, inv=new_inv, cell_dict=new_dict, 
                                        active=1, cells=new_cells, code=code)
    
    
    # print('finish_one_ref:', new_partition.cell_dict)
    return new_partition
    
def one_dimensional_refinement_2(G, partition, node_invariant):
    
    
    new_partition = one_dimensional_refinement(G, partition, node_invariant)
    # while not is_equitable(G, new_partition):
    
    #     new_partition = one_dimensional_refinement(G, partition, node_invariant)
        
    return new_partition



def is_equitable(G, partition):
    for cell in range(partition.cells):
        nodes = partition.cell_dict[cell]
        for node_i, node_j in itertools.combinations(nodes, 2):
            neighbor_i = G.neighbor_dict[node_i]
            neighbor_j = G.neighbor_dict[node_j]
            node_i_color = node_neighbors_color(neighbor_i, partition.cls)
            # print(f'here6: {node_i_color}')
            node_j_color = node_neighbors_color(neighbor_j, partition.cls)
            # print(f'here7: {node_j_color}')

            if node_i_color != node_j_color:
                return False
    return True



def k_dimensional_refinement(G, partition, generators, node_invariant):

    partition = one_dimensional_refinement(G, partition, node_invariant)
    queue = [partition]
    
    while queue:
        current_partition = queue.pop(0)
        # print(f'what! {current_partition.cls}')

        
        max_count = -1
        target_cell = -1
        current_partition = copy.deepcopy(current_partition)

        for idx, cell in current_partition.cell_dict.items():
            count = 0
            if len(cell) == 1:
                continue
            for node in cell:
                # print(f'current: ', current_partition.cls)
                # print(f'individulized node', node, cell)
                
                new_partition = individualize(current_partition, node)
                # print(current_partition.cell_dict)
                # print(new_partition.cell_dict)
                # sys.exit()
                new_partition = one_dimensional_refinement(G, new_partition, node_invariant)
                # print(new_partition.cls)
                # print(new_partition.cell_dict)
                if is_discrete(new_partition):
                    count += 1
                    update_generators(generators, G)
                else:
                    queue.append(new_partition)
            # print('shenme!', len(queue))
            if count > max_count:
                max_count = count
                target_cell = idx
        
    return new_partition

def is_discrete(partition):
    # print('haaaaaaaaaa', partition.cells, len(partition.cls))
    return partition.cells == len(partition.cls)


# def update_generators(generators, graph):
#     current_generators = [Permutation(p) for p in generators]
#     new_generators = random_schreier_sims(generators, graph.num_nodes)
#     generators.extend(new_generators)


def symmetry_pruning(traces, current_trace, generators):

    # for generator in generators:
    #     perm = Permutation(generator)
    #     if apply_automorphism(trace, current_trace, perm):
    #         return True
    
    for g in generators:
        for trace in traces:
            if are_sequences_equinvariant(trace, current_trace, g):
                return True
    return False

def apply_automorphism(trace, current_trace, generator):
    index_map = list(generator)
    permuted_sequence = [trace[index_map[i]] for i in range(len(trace))]
    return permuted_sequence == current_trace


def are_sequences_equinvariant(seq1, seq2, g):
    if len(seq1) != len(seq2):
        return False
    transformed_seq1 = apply_automorphism(seq1, g)
    return transformed_seq1 == seq2


def apply_automorphism(sequence, g):
    return tuple(g[v] for v in sequence)


    # new_cls = partition.cls[:]

    # for i in range(len(generator)):
    #     new_cls[i] = partition.cls[generator[i]]

    # new_partition = Partition(cls=new_cls, inv=partition.inv, cell_dict=partition.cell_dict, active=partition.active, cells=partition.cells, code=partition.code)
    # return new_partition



def invariant_pruning(partition, invariants, node_invariant):
    invariant_code = compute_invariant(partition, node_invariant)
    if invariant_code in invariants:
        return True
    invariants.add(invariant_code)
    return False


def compute_invariant(partition, node_invariant):
    return node_invariant.compute_invariant(partition)


def select_target_cell(partition, node_invariant):
    target_cell = None
    max_size = -1
    
    for key, value in partition.cell_dict.items():
        curr_size = len(value)
        if curr_size > max_size:
            target_cell = key
            max_size = curr_size
        elif curr_size == max_size:
            # print('here')
            curr_cell = node_invariant.node_each_cell_invariant(value, partition.cls)
            target_cell_items = node_invariant.node_each_cell_invariant(partition.cell_dict[target_cell], partition.cls)

            if curr_cell > target_cell_items:
                # print('here2')
                target_cell = key
                max_size = curr_size
            # elif curr_cell == target_cell_items:
            #     print('here!!')
            #     sys.exit()
    return target_cell


def prune_pa(nodes_to_refine, current_invariant):
    for node in nodes_to_refine:
        if node.partition.code < current_invariant:
            nodes_to_refine.remove(node)

def prune_pb(nodes_to_refine, current_invariant):
    for node in nodes_to_refine:
        if node.partition.code != current_invariant:
            nodes_to_refine.remove(node)


def find_node_largest_invariant(partition, target_cell, node_invariant):
    best_node = None
    largest = None
    best_partition = None
    for node in partition.cell_dict[target_cell]:
        current_partition = individualize(partition, node)
        curr_node_invariant = node_invariant.compute_invariant(current_partition)
        if largest is None or curr_node_invariant > largest:
            largest = curr_node_invariant
            best_node = node  
            best_partition = current_partition

    return best_node, best_partition




def draw_graph(graph, labels=None, title="Graph"):
    pos = nx.spring_layout(graph)  # positions for all nodes
    plt.figure() 

    nx.draw(graph, pos, with_labels=True, labels={node: node for node in graph.nodes()}, node_color='lightblue', node_size=500, font_size=10, font_color='black', edge_color='gray')
    if labels:
        labels_pos1 = {node: (pos[node][0], pos[node][1] + 0.05) for node in pos}
        nx.draw_networkx_labels(graph, labels_pos1, labels=labels, font_size=12, font_color='red')
    # ax.set_title(title1)

    plt.title(title)
    plt.show()



def refine_partition(graph, partition, node_invariant, seen_codes):
    
    
    partition = one_dimensional_refinement(graph, partition, node_invariant)
    print('At beginning')
    # print(partition.cls)
    print(partition.cell_dict)
    # partition.code = 
    a = {}
    for i in range(graph.num_nodes):
        a[i] = partition.cls[i]
    # draw_graph(graph.graph, a)
    
    traces = []
    trace = []
    root = TreeNode(partition, trace)
    current_level_nodes = [root]
    next_level_nodes = []
    leaves = []
    level = 0
    generators = []



    while current_level_nodes:
        print('New level', level, 'length:', len(current_level_nodes))
        # print()
        for current_node in current_level_nodes:
            # print(f'current level {level}, current node {current_level_nodes.index(current_node)}')
            current_partition = current_node.partition
            current_invariant = current_partition.code
            # print('here!!!!', len(current_node.sequence))
            if is_discrete(current_node.partition):
                # print('discrete node idx', current_level_nodes.index(current_node))
                traces.append(current_node.sequence)
                leaves.append(current_node)
                if len(leaves) == 2:
                    a, b = leaves[0], leaves[1]
                    schreier_sims = initialize_schreier_sims_with_partitions(a.partition, b.partition)
                    continue
                elif len(leaves) !=1:
                    schreier_sims.update_with_partition(current_partition)
                    generators = schreier_sims.get_generators()
                    # print('here', generators)
                    continue
                else:
                    continue

            if len(leaves) > 2:
                if symmetry_pruning(traces, current_node.sequence, generators):
                    continue
            
            prune_pa(current_level_nodes, current_invariant)
            prune_pb(current_level_nodes, current_invariant)


            if current_invariant not in seen_codes:
                seen_codes.add(current_invariant)

            target_cell = select_target_cell(current_partition, node_invariant)
            # print(f'target_cell: {target_cell}')
            a = node_invariant.node_each_cell_invariant(current_partition.cell_dict[target_cell], current_partition.cls)
            # print(a)
            for node in a:
                temp_trace = copy.deepcopy(current_node.sequence)
                temp_trace.append(node)
                indi_partition = individualize(current_partition, node)
                # print(f'indi node {node}')
                # print(f'after indi {indi_partition.cell_dict}')
                partition = one_dimensional_refinement(graph, indi_partition, node_invariant)
                # print(f'after refin {partition.cell_dict}')
                child_node = TreeNode(partition, parent=current_node, sequence=temp_trace)
                current_node.add_child(child_node)
                next_level_nodes.append(child_node)
            
        level += 1
        current_level_nodes = next_level_nodes
        next_level_nodes = []
    
    return root, leaves, generators

    