import collections

import numpy as np

from src.data.preprocessing.split_baseline import node_removal
from src.data.preprocessing.utils import remove_useless_node, GraphAdj, remove_loose_input


def compare_node_type(node_i, node_j):
    if len(node_i) != len(node_j):
        return False
    for _i, _j in zip(node_i, node_j):
        if _i != _j:
            return False
    return True


def encode_node_types(adj_matrix):
    num_nodes = adj_matrix.shape[0]

    node_types = []
    for _i in range(num_nodes):
        _n_type = np.where(adj_matrix[:, _i] != 0)[0]
        _n_type.sort()
        node_types.append(_i - _n_type)

    type_of_nodes = {}
    node_types_idxs = []
    _curr_max_type = 0

    for n_i in node_types:
        is_in_type = False
        type_idx = None

        for n_j in type_of_nodes:
            if compare_node_type(n_i, type_of_nodes[n_j]):
                is_in_type = True
                type_idx = n_j
                break

        if not is_in_type:
            type_of_nodes[_curr_max_type] = n_i
            type_idx = _curr_max_type
            _curr_max_type += 1

        node_types_idxs.append(type_idx)

    return type_of_nodes, node_types_idxs


def find_repeat_subseqs(seq, init_subseq_len=2, threshold=None):
    count_subseq = collections.OrderedDict()

    for i in range(len(seq) - (init_subseq_len - 1)):
        subseq_str = str(seq[i:i + init_subseq_len])

        count_subseq[subseq_str] = count_subseq.get(subseq_str, 0) + 1

    freqs = []
    for i in range(len(seq) - 1):
        subseq_str = str(seq[i:i + init_subseq_len])

        freqs.append(count_subseq[subseq_str])

    # print(freqs)

    if threshold is None:
        # auto set threshold
        diffs = []
        for i in range(len(freqs) - 1):
            diffs.append(abs(freqs[i + 1] - freqs[i]))

        diffs_count = {i: diffs.count(i) for i in set(diffs) if i != 0}
        if len(diffs_count) != 0:
            max_diff = -1
            max_key = None
            for k in diffs_count:
                if diffs_count[k] > max_diff:
                    max_diff = diffs_count[k]
                    max_key = k
            threshold = max_key
        else:
            threshold = 0

    subseqs = [[seq[0]]]
    subseq_i = 0
    i = 0
    while i < len(freqs) - 1:
        if abs(freqs[i] - freqs[i + 1]) <= threshold:
            subseqs[subseq_i].append(seq[i + 1])
            i += 1
        else:
            if freqs[i] > freqs[i + 1]:
                subseqs[subseq_i].append(seq[i + 1])
                i += 1

            subseq_i += 1
            subseqs.append([seq[i + 1]])
            i += 1

    if i + 1 < len(seq):
        subseqs[subseq_i].append(seq[i + 1])

    return subseqs


def find_frequent_motifs(graph_edges, with_remove=True):

    if with_remove:
        graph_edges = node_removal(graph_edges=graph_edges)

    graph = GraphAdj(graph_edges=graph_edges)

    type_of_nodes, node_types_idxs = encode_node_types(graph.adj_matrix)

    subseqs = find_repeat_subseqs(node_types_idxs)

    lens_subseqs = [len(ss) for ss in subseqs]

    i = 0
    list_of_subgraph_node = []
    for l in lens_subseqs:
        list_of_subgraph_node.append(graph.nodes_list[i:i + l])
        i += l

    list_of_subgraph_edges = []
    for i, ssn in enumerate(list_of_subgraph_node):
        sg_edges = []
        for node_i, node_j in graph_edges:
            if node_i in ssn or node_j in ssn:  # TODO: `and` OR `or`?
                sg_edges.append((node_i, node_j))
        list_of_subgraph_edges.append(sg_edges)

    return list_of_subgraph_edges