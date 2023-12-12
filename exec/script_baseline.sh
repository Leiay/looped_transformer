
n_gpu=0

# Linear Regression
#python scripts/train.py --config configs/base.yaml \
#    --wandb.name "LR_baseline" \
#    --gpu.n_gpu $n_gpu

## Sparse LR
#python scripts/train.py --config configs/sparse_LR/base.yaml \
#    --wandb.name "SparseLR_baseline" \
#    --gpu.n_gpu $n_gpu
#
## Decision Tree
#python scripts/train.py --config configs/decision_tree/base.yaml \
#    --wandb.name "DT_baseline" \
#    --gpu.n_gpu $n_gpu

## ReLU 2NN
#python scripts/train.py --config configs/relu_2nn_regression/base.yaml \
#    --wandb.name "ReLU2NN_baseline" \
#    --gpu.n_gpu $n_gpu
