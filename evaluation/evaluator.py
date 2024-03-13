import re
import os
import sys
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import pickle
from src.config import Config
import json
from src.data.dataset.nas.darts import get_unique_nodes, encode_edges, name2idx
import torch
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_all_paths(root):
    paths = os.listdir(root)
    paths = [os.path.join(root, p) for p in paths]
    return paths

def load_from_file(path, mode='rb'):
    # print(path)
    mode = 'rb' if path.endswith('.pkl') else 'r'
    if mode == 'rb':
        with open(path, mode) as fp:
            data = pickle.load(fp)
    else:
        with open(path, mode) as fp:
            data = json.load(fp)
    return data


import numpy as np
# import torch
import faiss
import pickle


def evaluator(ground_truth, rank_index, top_k=10, metrics=['mrr', 'map', 'ndcg'], use_graded_scores=False):
    # print(ground_truth)
    # print(rank_index[0])
    # rank_index = rank_index[0]
    results = {}

    if 'mrr' in metrics:
        mrr = 0.
        for rank, item in enumerate(rank_index[:top_k]):
            # print(type(item), item)
            if item in ground_truth:
                mrr = 1.0 / (rank + 1.0)
                break
        results['mrr@'+str(top_k)] = mrr

    if 'map' in metrics:
        if not ground_truth:
            return 0.
        map = 0.
        num_hits = 0.
        for rank, item in enumerate(rank_index[:top_k]):
            if item in ground_truth and item not in rank_index[:rank]:
                num_hits += 1.
                map += num_hits / (rank + 1.0)
        map = map / max(1.0, len(ground_truth))

        results['map@'+str(top_k)] = map
    
    if 'ndcg' in metrics:
        ndcg = 0.
        for rank, item in enumerate(rank_index[:top_k]):
            if item in ground_truth:
                if use_graded_scores:
                    grade = 1.0 / (ground_truth.index(item) + 1)
                else:
                    grade = 1.0
                ndcg += grade / np.log2(rank + 2)

        norm = 0.0
        for rank in range(len(ground_truth)):
            if use_graded_scores:
                grade = 1.0 / (rank + 1)
            else:
                grade = 1.0
            norm += grade / np.log2(rank + 2)
        results['ndcg@'+str(top_k)] = ndcg / max(0.3, norm)

    return results

def get_k_neighbors(query: np.ndarray, database: np.ndarray, k: int):
    
    ngpus = faiss.get_num_gpus()
    
    assert query.shape[-1] == database.shape[-1]

    dimention = query.shape[-1]
    # print(ngpus)
    if ngpus == 0:
        index = faiss.IndexFlatL2(dimention)
        index.add(database)
        D, I = index.search(query, k)     # actual search
        
    elif ngpus == 1:
        res = faiss.StandardGpuResources()  # use a single GPU
        ## Using a flat index
        index_flat = faiss.IndexFlatL2(dimention)  # build a flat (CPU) index
        # make it a flat GPU index
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

        gpu_index_flat.add(database)         # add vectors to the index

        D, I = gpu_index_flat.search(query, k)  # actual search

    else:
        cpu_index = faiss.IndexFlatL2(database)
        gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
            cpu_index
        )
        gpu_index.add(database)              # add vectors to the index
        D, I = gpu_index.search(query, k) # actual search

    return I[:k]


def get_ground_truth(model_name: str, labels2idx=None, class2truth=None): # num_class
    
    class_num = name2idx(model_name, labels2idx)
    ground_truth_idx = class2truth[class_num]

    return set(ground_truth_idx)


def get_scores(query, database, ground_truth, top_k=10):

    rank_index = get_k_neighbors(query, database, top_k)

    results = evaluator(ground_truth, rank_index, top_k=top_k, metrics=['mrr', 'map', 'ndcg'])
    return results

def search(query, database, top_k=10):
    top_k_index =  get_k_neighbors(query, database, k=top_k)
    top_k_emb= database[top_k_index, :]
    return top_k_index, top_k_emb

def main():

    config = Config(default_config_file='configs/name2truth.yaml') 
    args = config.load_config()
    # unique_names = set()
    file_name = os.path.join(args.output_root, 'labels2idx.json')
    with open(file_name, 'r') as f:
        labels2idx = json.load(f)
    file_name = os.path.join(args.output_root, 'class2truth.json')
    with open(file_name, 'r') as f:
        class2truth = json.load(f)

    pretrained_emb = torch.load('checkpoints/DartsPretraining-Exp/embeddings.pt')

    paths_ = get_all_paths(args.root)
    models = [file for file in paths_ if re.search('.*\.json$', file)]

    top_k = 10
    results = {'mrr@'+str(top_k): 0., 'map@'+str(top_k): 0., 'ndcg@'+str(top_k): 0.}
    
    for idx, model in enumerate(models):
        # print(data['model_name'])
        model_name = model.split('/')[-1][6:-5]
        # print(model_name)
        # exit(0)
        model_class = name2idx(model_name, labels2idx)
        ground_truth = class2truth[str(model_class)]
        ground_truth = set(ground_truth)
        query_emb = pretrained_emb[idx].reshape(1, -1).numpy().astype('float32')
        sim_index = get_k_neighbors(query_emb, pretrained_emb, top_k)[0]
        
        rank_index = sim_index[sim_index != idx]
        results_ = evaluator(ground_truth, rank_index, top_k=top_k, metrics=['mrr', 'map', 'ndcg'])


        results['mrr@'+str(top_k)] += results_['mrr@'+str(top_k)]
        results['map@'+str(top_k)] += results_['map@'+str(top_k)]
        results['ndcg@'+str(top_k)] += results_['ndcg@'+str(top_k)]

    results['mrr@'+str(top_k)] = results['mrr@'+str(top_k)] / len(models)
    results['map@'+str(top_k)] = results['map@'+str(top_k)] / len(models)
    results['ndcg@'+str(top_k)] = results['ndcg@'+str(top_k)] / len(models)
    print(results)
    
if __name__ == '__main__':
    main()
    