from two_dim_ref import refine_partition, is_discrete, apply_automorphism
# from SS_al import random_schreier_sims
from utils import TracesVars, initialize_part, Candidate, NodeInvariant, TreeNode, LabeledGraph
import sys
import networkx as nx
import random
import matplotlib.pyplot as plt
import re

def create_quotient_graph(G, partition):
    Q = nx.Graph()

    for cell_id, nodes in partition.cell_dict.items():
        Q.add_node(cell_id, size=len(nodes), old_label=nodes)

    for cell_id1, nodes1 in partition.cell_dict.items():
        for cell_id2 in partition.cell_dict:
            if cell_id1 <= cell_id2:
                weight = sum(1 for node1 in nodes1 for node2 in partition.cell_dict[cell_id2] if G.graph.has_edge(node1, node2))
                if weight > 0:
                    Q.add_edge(cell_id1, cell_id2, weight=weight)

    return Q


def compute_canonical_form(graph, leaves):
    best_labeling = None
    best_code = None
    

    for node in leaves:
        # print(node.sequence)
        current_partition = node.partition
        # print(current_partition.cls)
        # print(current_partition.code)
        if best_code is None or current_partition.code > best_code:
            best_code = current_partition.code
            best_labeling = current_partition.cls
            best_partition = current_partition
    
    # Q = create_quotient_graph(graph, best_partition)
    # canonical_form = nx.convert_node_labels_to_integers(Q, label_attribute="old_label")
    # sorted_nodes = sorted(canonical_form.nodes(data=True), key=lambda x: (x[1]['old_label'], x[0]))
    
    # canonical_labels = {data['old_label']: idx for idx, (node, data) in enumerate(sorted_nodes)}
    # print(best_labeling)
    return best_labeling



def draw_two_graphs(graph1, graph2, labels1=None, labels2=None, title1="Graph 1", title2="Graph 2"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    pos1 = nx.spring_layout(graph1)
    pos2 = nx.spring_layout(graph2)

    nx.draw(graph1, pos1, with_labels=True, labels={node: node for node in graph1.nodes()}, node_color='lightblue', node_size=500, font_size=10, font_color='black', edge_color='gray', ax=ax1)
    if labels1:
        labels_pos1 = {node: (pos1[node][0], pos1[node][1] + 0.1) for node in pos1}
        nx.draw_networkx_labels(graph1, labels_pos1, labels=labels1, font_size=12, font_color='red', ax=ax1)
    ax1.set_title(title1)

    nx.draw(graph2, pos2, with_labels=True, labels={node: node for node in graph2.nodes()}, node_color='lightgreen', node_size=500, font_size=10, font_color='black', edge_color='gray', ax=ax2)
    if labels2:
        labels_pos2 = {node: (pos2[node][0], pos2[node][1] + 0.1) for node in pos2}
        nx.draw_networkx_labels(graph2, labels_pos2, labels=labels2, font_size=12, font_color='red', ax=ax2)
    ax2.set_title(title2)

    plt.show()


def canonical_form_graph(graph, canonical_label):
    
    
    canonical_graph = nx.Graph()
    

    canonical_graph.add_nodes_from(range(len(canonical_label)))
    
    for u, v in graph.graph.edges():
        canonical_graph.add_edge(canonical_label[u], canonical_label[v])
    
    return canonical_graph



def check_iso(form1, form2, graph1, graph2):
    for u, v in graph1.edges():
        node1 = form2.index(form1[u])
        node2 = form2.index(form1[v])
        if (node1, node2) in graph2.edges():
            continue
        else:
            return False

    return True



def traces(graph_file, options, stats, dataset):

    graph = LabeledGraph(graph_file, dataset)
    vars = TracesVars(graph, options, stats)
    vars.partition = initialize_part(graph)
    vars.node_invariant = NodeInvariant(graph)
    invariants = set()
    root, leaves, generators = refine_partition(graph, vars.partition, vars.node_invariant, vars.seen_codes)
    if not root:
        return None
    vars.canonical_form = compute_canonical_form(vars.graph, leaves)
    return vars, generators



################################################################################################################################


# def one_run(group_id, sample_id):
#     options = None
#     stats = None
#     vars, generators = traces(f'/home/cds/Documents/AAAI25/canonical_labeling/syn_data_2/{group_id}/{sample_id}.npy', options, stats)
#     return vars.canonical_form, vars.graph.graph

# def one_group_run(group_id):
#     current_form, current_graph = one_run(group_id, 0)
#     for i in range(1, 2):
#         # print(i)
#         form, graph = one_run(group_id, i)
#         if not check_iso(current_form, form, current_graph, graph):
#             print(f'Group{group_id}: Error')
            
#             print(current_form)
#             print(form)
#             current_label = {}
#             label = {}
#             for i in range(len(current_form)):
#                 current_label[i] = current_form[i]
#                 label[i] = form[i]

#             # draw_two_graphs(current_graph, graph, current_label, label)
#             return False

#     print(f'Group{group_id}: Success!')
#     # print(f'Cananical Form: {current_form}')
#     return True


# def non_iso_graphs(group_id1, group_id2):
#     sample_id_1 = random.randint(1, 9)
#     sample_id_2 = random.randint(1, 9)

#     form1 = one_run(group_id1, sample_id_1)
#     form2 = one_run(group_id2, sample_id_2)

#     if form1 != form2:
#         print('Correct decision')
#         # print(form1)
#         # print(form2)
#     else:
#         print('Wrong')
#         # print(form1)
#         # print(form2)

# print('====================================== isomorphism test')

# for i in range(1, 100):
#     a = one_group_run(i)
# print('====================================== isomorphism test')
# a = one_group_run(1)
# print('====================================== non isomorphism test')
# non_iso_graphs(1, 2)


################################################################################################################################


# def modify_filename(file_path):
#     parts = file_path.rsplit('-', 1)  # Split the file path at the last hyphen
#     if len(parts) == 2 and parts[1].isdigit():
#         new_file_path = f"{parts[0]}-{int(parts[1]) + 1}"
#         return new_file_path
#     return file_path

# def modify_filename_with_path(file_path):
#     path_parts = file_path.rsplit('/', 1)
#     if len(path_parts) == 2:
#         modified_filename = modify_filename(path_parts[1])
#         new_file_path = f"{path_parts[0]}/{modified_filename}"
#         return new_file_path
#     return file_path

# def benchmark1_run_helper(file_path):
#     options = None
#     stats = None
#     vars, generators = traces(file_path, options, stats, dataset='benchmark1')
#     return vars.canonical_form, vars.graph.graph

# def benchmark1_run(file_path):
#     current_form, current_graph = benchmark1_run_helper(file_path)
#     next_file_path = modify_filename_with_path(file_path)
#     print(next_file_path)
#     form, graph = benchmark1_run_helper(next_file_path)
#     if not check_iso(current_form, form, current_graph, graph):
#         print(f'Error')
        
#         # print(current_form)
#         # print(form)
#         current_label = {}
#         label = {}
#         for i in range(len(current_form)):
#             current_label[i] = current_form[i]
#             label[i] = form[i]
        
#         # draw_two_graphs(current_graph, graph, current_label, label)
        
#         return False

#     print(f'Success!')
#     # print(f'Cananical Form: {current_form}')
#     return True


# a = benchmark1_run('/home/cds/Documents/AAAI25/canonical_labeling/benchmark1/cfi-rigid-t2/cfi-rigid-t2-0016-04-1')


################################################################################################################################



# def regular_run():
#     def one_run(file_path):
#         options = None
#         stats = None
#         vars, generators = traces(file_path, options, stats)
#         return vars.canonical_form, vars.graph.graph
    
#     current_form, current_graph = one_run('/home/cds/Documents/AAAI25/canonical_labeling/graph-1.npy')
#     form, graph = one_run('/home/cds/Documents/AAAI25/canonical_labeling/graph-2.npy')
#     if not check_iso(current_form, form, current_graph, graph):
#         print(f'Error')
        
#         # print(current_form)
#         # print(form)
#         current_label = {}
#         label = {}
#         for i in range(len(current_form)):
#             current_label[i] = current_form[i]
#             label[i] = form[i]
        
#         # draw_two_graphs(current_graph, graph, current_label, label)
        
#         return False

#     print(f'Success!')
#     # print(f'Cananical Form: {current_form}')
#     return True

################################################################################################################################


class main_run:
    def __init__(self, dataset):
        self.dataset = dataset

    def one_graph_run(self, file_path):
        options = None
        stats = None
        vars, generators = traces(file_path, options, stats, dataset=self.dataset)
        return vars.canonical_form, vars.graph.graph
    
    def modify_filename(self, file_path):
        parts = file_path.rsplit('-', 1)  # Split the file path at the last hyphen
        if len(parts) == 2 and parts[1].isdigit():
            new_file_path = f"{parts[0]}-{int(parts[1]) + 1}"
            return new_file_path
        return file_path

    def modify_filename_with_path(self, file_path):
        if self.dataset == 'benchmark1':
            path_parts = file_path.rsplit('/', 1)
            if len(path_parts) == 2:
                modified_filename = self.modify_filename(path_parts[1])
                new_file_path = f"{path_parts[0]}/{modified_filename}"
                return new_file_path
            return file_path
        elif self.dataset == 'data_from_software':
            if re.search(r'-1\.dre$', file_path):
                output_string = re.sub(r'-1\.dre$', '-2.dre', file_path)
            elif re.search(r'_a\.dre$', file_path):
                output_string = re.sub(r'_a\.dre$', '_b.dre', file_path)
            else:
                output_string = file_path  # If no match, return the input string unchanged
            
            return output_string



    def true_iso_graphs_run(self, file_path):
        current_form, current_graph = self.one_graph_run(file_path)
        next_file_path = self.modify_filename_with_path(file_path)
        print(next_file_path)
        form, graph = self.one_graph_run(next_file_path)
        if not check_iso(current_form, form, current_graph, graph):
            print(f'Error')
            
            # print(current_form)
            # print(form)
            current_label = {}
            label = {}
            for i in range(len(current_form)):
                current_label[i] = current_form[i]
                label[i] = form[i]
            
            # draw_two_graphs(current_graph, graph, current_label, label)
            
            return False

        print(f'Success!')
        # print(f'Cananical Form: {current_form}')
        return True
    

dataset = 'data_from_software'
# dataset = 'benchmark1'
run = main_run(dataset)
a = run.true_iso_graphs_run('/home/cds/Documents/AAAI25/canonical_labeling/data/data_from_software/sat_cfi_dre/sat_cfi_base_0100_a.dre')
