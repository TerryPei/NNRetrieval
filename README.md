# # Introduction For Neural Architecture Retrieval

#### Code implementation of NAR.

![LICENSE](https://img.shields.io/github/license/MAR)
![VERSION](https://img.shields.io/badge/version-v1.01-blue)
![PYTHON](https://img.shields.io/badge/python-3.9.2-orange)
![MODEL](https://img.shields.io/badge/NAR)

# Install

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
    --device 'cpu'
```

Generation: 10,000 sample / 7 sec (118 MB)
- 3.28 min/ep @ bs=6
- bs=512, GPU Mem=8.95GB
- 3.15 min/ep bs=1,024, GPU Mem=20.75GB
