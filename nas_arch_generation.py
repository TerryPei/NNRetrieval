import csv
import json
import os

import numpy as np
from tqdm import tqdm

from src.config import Config
from src.data.dataset.nas.darts import get_unique_nodes, encode_edges
from src.data.dataset.nas.generation.darts import DartsCellArch, DartsNetworkArch
from src.data.preprocessing.utils import adj_matrix2edge_list


def darts_generation(num_arch, num_inter_nodes=4, shuffle=True):
    cell_arch = DartsCellArch(num_inter_nodes=num_inter_nodes)
    total_num = cell_arch.num_all_op_combs * cell_arch.num_all_edge_combs

    architectures = {'normal': [], 'reduction': []}  # two kinds of cells

    for k in architectures:

        # randomly sample architecture indices
        idx_samples = np.array([], dtype=int)
        while idx_samples.shape[0] < num_arch:
            new_samples = np.random.randint(low=0, high=total_num, size=num_arch - idx_samples.shape[0], dtype=int)
            idx_samples = np.unique(np.concatenate([idx_samples, new_samples]))

        # shuffle indices
        if shuffle:
            np.random.shuffle(idx_samples)

        # sample architectures based on indices
        for arch_idx in idx_samples:
            arch = cell_arch.sample_arch(arch_idx)
            edges_list = adj_matrix2edge_list(
                adj_matrix=arch['adj_matrix'],
                node_list=arch['node_list']
            )

            architectures[k].append({
                'edges_list': edges_list,
                'arch_idx': arch_idx,
            })

    # build network graph based on sampled architectures
    # network_graphs = []
    for normal_arch, reduction_arch in zip(architectures['normal'], architectures['reduction']):
        network = DartsNetworkArch(
            normal_arch=normal_arch['edges_list'],
            reduction_arch=reduction_arch['edges_list']
        )

        # network_graphs.append(network.edges_list)
        yield normal_arch['arch_idx'], reduction_arch['arch_idx'], network.edges_list

    # return network_graphs


def main():
    config = Config()
    config.parser.add_argument('--output', type=str, default='.')
    config.parser.add_argument('--num_arch', type=int, default=100)
    config.parser.add_argument('--format', type=str, default='json', choices=['csv', 'json'])
    args = config.load_config()

    network_graphs_iter = darts_generation(num_arch=args.num_arch, num_inter_nodes=4)

    if 'csv' == args.format:
        header = ['graph_index', 'node_i', 'node_j']
        path = os.path.join(
            args.output,
            'darts-%d.csv' % args.num_arch
        )
        with open(path, 'w') as fp:
            writer = csv.writer(fp)
            writer.writerow(header)
            for i, g in enumerate(network_graphs_iter):
                for node_i, node_j in g[-1]:
                    writer.writerow([i, node_i, node_j])
    elif 'json' == args.format:
        path = os.path.join(
            args.output,
            'darts-json-%d' % args.num_arch,
        )
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            raise RuntimeError('%s exists' % path)
        with tqdm(total=args.num_arch) as p_bar:
            p_bar.set_description('Saving to JSON:')
            for norm_arch_idx, red_arch_idx, edges_list in network_graphs_iter:
                # get unique nodes
                unique_nodes = get_unique_nodes(edges=edges_list)
                # mapping nodes to index
                node2idx = {node: idx for idx, node in enumerate(unique_nodes)}
                # encode edges
                edge_index = encode_edges(edges=edges_list, node2idx=node2idx)
                # dump as json
                g_json = json.dumps({
                    'unique_nodes': list(unique_nodes),
                    'edge_index': edge_index.tolist(),
                })
                # save to files
                f_name = os.path.join(path, 'graph-%012d-%012d.json' % (norm_arch_idx, red_arch_idx))
                with open(f_name, 'w') as fp:
                    fp.write(g_json)

                p_bar.update(1)


if __name__ == '__main__':
    main()
