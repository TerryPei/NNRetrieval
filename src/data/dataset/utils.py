from src.data.preprocessing.frequent_motif import find_frequent_motifs
# from src.data.preprocessing.split_baseline import split_by_node_num, split_by_motif_num, split_random
from src.data.preprocessing.split_baseline import get_split_by_node_num, get_split_random
import numpy as np
import copy

# split_func = find_frequent_motifs

# Baseline 1
# split_func = get_split_by_node_num(12345)

# Baseline 2
split_func = get_split_random(5, 10)


def build_subgraph_context_pairs(
        graph_edges,
        split_func=split_func,
        with_remove=True
):

    list_of_subgraph_edges = split_func(
        graph_edges=graph_edges, with_remove=with_remove
    )
    
# def build_subgraph_context_pairs(graph_edges, with_remove=True):

#     list_of_subgraph_edges = find_frequent_motifs(
#         graph_edges=graph_edges, with_remove=with_remove, 
#     )

    pairs = {
        'subgraph': list_of_subgraph_edges,
        'pos_context': [],
        'neg_context': None,
    }

    for sg_edges in list_of_subgraph_edges:
        # get unique nodes
        sg_nodes = []
        for node_i, node_j in sg_edges:
            sg_nodes.append(node_i)
            sg_nodes.append(node_j)
        sg_nodes = set(sg_nodes)

        # find order-1 context, TODO: other orders
        c_edges= []
        for node_i, node_j in graph_edges:
            if node_i in sg_nodes or node_j in sg_nodes:
                c_edges.append((node_i, node_j))

        # save as positive pairs
        pairs['pos_context'].append(c_edges)
    # # TODO: other methods to get **random** neg_pairs
    # split = len(pairs['pos_context']) // 2
    # pairs['neg_context'] = pairs['pos_context'][-split:] + pairs['pos_context'][:-split]
    if pairs['pos_context'][0] == [] and len(pairs['pos_context']) > 1:
        pairs['pos_context'][0] = pairs['pos_context'][1]
    # return pairs
    # TODO: other methods to get **random** neg_pairs
    # neg_context = copy.deepcopy(pairs['pos_context'])
    # np.random.shuffle(neg_context)
    # pairs['neg_context'] = neg_context
    split = len(pairs['pos_context']) // 2
    pairs['neg_context'] = pairs['pos_context'][-split:] + pairs['pos_context'][:-split]
    return pairs

# def build_subgraph_context_pairs(
#         graph_edges,
#         split_func=find_frequent_motifs,
#         with_remove=True
# ):

#     list_of_subgraph_edges = split_func(
#         graph_edges=graph_edges, with_remove=with_remove
#     )

#     pairs = {
#         'subgraph': list_of_subgraph_edges,
#         'pos_context': [],
#         'neg_context': [],
#     }

#     for sg_edges in list_of_subgraph_edges:
#         # get unique nodes
#         sg_nodes = []
#         for node_i, node_j in sg_edges:
#             sg_nodes.append(node_i)
#             sg_nodes.append(node_j)
#         sg_nodes = set(sg_nodes)

#         # find order-1 context, TODO: other orders
#         c_edges= []
#         for node_i, node_j in graph_edges:
#             if node_i in sg_nodes or node_j in sg_nodes:
#                 c_edges.append((node_i, node_j))

#         # save as positive pairs
#         pairs['pos_context'].append(c_edges)

#     # get **random** neg_pairs
#     neg_context = copy.deepcopy(pairs['pos_context'])

#     indices_pos = np.arange(len(neg_context))
#     indices_neg = np.arange(len(neg_context))
#     num_loop = 0
#     while (indices_pos == indices_neg).any():
#         indices_to_be_shuffle = np.where(indices_pos == indices_neg)[0]
#         if indices_to_be_shuffle.shape[0] < 2:
#             if indices_to_be_shuffle <= len(neg_context) - 2:
#                 indices_to_be_shuffle = np.concatenate([
#                     indices_to_be_shuffle,
#                     indices_to_be_shuffle + 1
#                 ])
#             elif indices_to_be_shuffle >= 1:
#                 indices_to_be_shuffle = np.concatenate([
#                     indices_to_be_shuffle,
#                     indices_to_be_shuffle - 1
#                 ])
#         tmp = indices_neg[indices_to_be_shuffle]
#         np.random.shuffle(tmp)
#         indices_neg[indices_to_be_shuffle] = tmp
#         num_loop += 1
#         if num_loop > 10:
#             # print(indices_pos == indices_neg)
#             break

#     for i in indices_neg:
#         pairs['neg_context'].append(neg_context[i])

#     return pairs

def build_macro_graph(list_subgraph, n_type=int):

    macro_arch_graph = []

    # compare each two subgraphs
    for idx_i, sg_i in enumerate(list_subgraph):
        for idx_j, sg_j in enumerate(list_subgraph[idx_i+1:]):
            idx_j += idx_i + 1
            # compare each two nodes in subgraph
            for n_ii, n_ij in sg_i:
                is_match = False
                for n_ji, n_jj in sg_j:
                    # if match
                    if n_ii == n_ji and n_ij == n_jj:
                        macro_arch_graph.append((n_type(idx_i), n_type(idx_j)))
                        is_match = True
                        break
                if is_match:
                    break

    return macro_arch_graph
