# ICLR 2024: Neural Architecture Retrieval (NAR)
#### Code implementation of our Accepted Paper: Neural Architecture Retrieval.

![LICENSE](https://img.shields.io/github/license/TerryPei/NAR)
![VERSION](https://img.shields.io/badge/version-v1.01-blue)
![PYTHON](https://img.shields.io/badge/python-3.9.2-orange)

<!-- ![MODEL](https://img.shields.io/badge/NAR) -->

<!-- ## Poster

<p align="center">
        <img src="results/figs/poster.png" width="460"/></a>
</p> -->


# Install `pytorch-geometric`

```shell script
TORCH=`python -c "import torch; print(torch.__version__)"` &&
CUDA=`python -c "import torch; print(torch.version.cuda)"`  &&
echo "TORCH=${TORCH}" &&
echo "CUDA=${CUDA}" &&
pip install torch-scatter -f "https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html" &&
pip install torch-sparse -f "https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html" &&
pip install torch-cluster -f "https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html" &&
pip install torch-spline-conv -f "https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html" &&
pip install torch-geometric
``` 


# RUN

## DARTS Data Generation

```shell script
python nas_arch_generation.py \
    --output data \
    --num_arch 10000
```

## DARTS Pre-training

```shell script
python run_pretraining.py \
    --config configs/darts_pretraining.yaml \
    --dataset_graph_path 'data/darts-json-20000' \
    --device 'gpu'
```

Generation: 10,000 sample / 7 sec (118 MB)
- 3.28 min/ep @ bs=6
- bs=512, GPU Mem=8.95GB
- 3.15 min/ep bs=1,024, GPU Mem=20.75GB

# Dataset

**Real world Dataset**

Download Link: [Real World Computational Graphs](https://drive.google.com/drive/folders/10bIbDNq4GqLNFGIkYD0swV8kVwLxJcYN?usp=drive_link).

**NAS Dataset**

```
cd .
python nas_arch_generation.py
```

## Results
### Table 1: Comparison with baselines on real-world neural architectures and NAS data.

| Dataset | Method | MRR           | MAP           | NDCG          |MRR           | MAP           | NDCG          |MRR           | MAP           | NDCG          |
|---------|--------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
|         |        | Top-20 | Top-50 | Top-100 | Top-20 | Top-50 | Top-100 | Top-20 | Top-50 | Top-100 |
| Real    | GCN    | 0.737  | 0.745  | 0.774   | 0.598  | 0.560  | 0.510   | 0.686  | 0.672  | 0.628   |
|         | GAT    | 0.756  | 0.776  | 0.787   | 0.542  | 0.541  | 0.538   | 0.610  | 0.598  | 0.511   |
|         | Ours   | 0.825  | 0.826  | 0.826   | 0.593  | 0.577  | 0.545   | 0.705  | 0.692  | 0.678   |
| NAS     | GCN    | 1.000  | 1.000  | 1.000   | 0.927  | 0.854  | 0.858   | 0.953  | 0.902  | 0.906   |
|         | GAT    | 1.000  | 1.000  | 1.000   | 0.941  | 0.899  | 0.901   | 0.961  | 0.933  | 0.935   |
|         | Ours   | 1.000  | 1.000  | 1.000   | 0.952  | 0.932  | 0.935   | 0.969  | 0.960  | 0.958   |

### Table 2: Evaluation of different graph split methods on real-world and NAS architectures.

| Dataset | Splitting | MRR           | MAP           | NDCG          | MRR           | MAP           | NDCG          | MRR           | MAP           | NDCG          |
|---------|-----------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
|         |           | Top-20 | Top-50 | Top-100 | Top-20 | Top-50 | Top-100 | Top-20 | Top-50 | Top-100 |
| Real    | Node Num  | 0.807  | 0.809  | 0.809   | 0.551  | 0.539  | 0.537   | 0.694  | 0.682  | 0.667   |
|         | Motif Num | 0.817  | 0.820  | 0.823   | 0.591  | 0.522  | 0.518   | 0.692  | 0.669  | 0.661   |
|         | Random    | 0.801  | 0.802  | 0.804   | 0.589  | 0.543  | 0.536   | 0.699  | 0.675  | 0.668   |
|         | Ours      | 0.825  | 0.826  | 0.826   | 0.593  | 0.577  | 0.545   | 0.705  | 0.692  | 0.678   |
| NAS     | Node Num  | 0.999  | 0.999  | 0.999   | 0.941  | 0.885  | 0.883   | 0.962  | 0.926  | 0.924   |
|         | Motif Num | 0.998  | 0.998  | 0.998   | 0.931  | 0.872  | 0.874   | 0.956  | 0.917  | 0.919   |
|         | Random    | 1.000  | 1.000  | 1.000   | 0.919  | 0.826  | 0.824   | 0.949  | 0.881  | 0.883   |
|         | Ours      | 1.000  | 1.000  | 1.000   | 0.952  | 0.936  | 0.935   | 0.969  | 0.957  | 0.958   |



## Playground Demo of Search Engine:

> release soon