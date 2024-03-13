import numpy as np

def evaluator(ground_truth, rank_index, top_k, metrics=['mrr', 'map', 'ndcg'], use_graded_scores=False):
    if len(rank_index) > top_k:
        rank_index = rank_index[:top_k]
    # if top_k > len(ground_truth):
    #     rank_index = rank_index[:len(ground_truth)]
    results = {}

    if 'mrr' in metrics:
        mrr = 0.
        for rank, item in enumerate(rank_index):
            # print(type(item), item)
            if item in ground_truth:
                mrr = 1.0 / (rank + 1.0)
                break
        results['mrr@'+str(top_k)] = mrr

    if 'map' in metrics:
        map = 0.
        num_hits = 0.
        for rank, item in enumerate(rank_index):
            if item in ground_truth and item not in rank_index[:rank]:
                num_hits += 1.
                map += num_hits / (rank + 1.0)
        if not ground_truth:
            results['map@'+str(top_k)] = 0.
        else:
            map = map / min(len(ground_truth), top_k)
            results['map@'+str(top_k)] = map
    
    if 'ndcg' in metrics:
        ndcg = 0.
        for rank, item in enumerate(rank_index):
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
        # print(ndcg, norm, max(0.3, norm))
        ndcg = ndcg / max(0.3, norm)

        results['ndcg@'+str(top_k)] = ndcg  

    return results

def main():

    ground_truth = {1, 2, 3, 9}
    predicted_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    top_k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for top_k in top_k_list:
        results_ = evaluator(ground_truth, predicted_index, top_k=top_k, metrics=['mrr', 'map', 'ndcg'])
        print(results_)
    
if __name__ == '__main__':
    main()
    