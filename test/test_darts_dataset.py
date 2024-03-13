from src.data.dataset.nas.darts import DartsSubgraphDatasetFolder


def test_subgraph_folder():
    darts_subgraph_dataset_folder = DartsSubgraphDatasetFolder('data/darts-json-1000')
    print('len:', len(darts_subgraph_dataset_folder))
    batch, macro_arch_graph = darts_subgraph_dataset_folder[0]
    print(batch)
    print(macro_arch_graph)
    batch, macro_arch_graph = darts_subgraph_dataset_folder[len(darts_subgraph_dataset_folder) - 1]
    print(batch)
    print(macro_arch_graph)


if __name__ == '__main__':
    test_subgraph_folder()
