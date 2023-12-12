
n_gpu=0

# Linear Regression  ###################################################################################################
b=30
T=15
#python scripts/train.py --config configs/base_loop.yaml \
#    --model.n_layer 1 \
#    --training.curriculum.loops.start $T \
#    --training.curriculum.loops.end $b \
#    --training.n_loop_window $T \
#    --wandb.name "LR_loop_L1_ends{$b}_T{$T}" \
#    --gpu.n_gpu $n_gpu

# Sparse Linear Regression  ############################################################################################
b=20
T=10
#python scripts/train.py --config configs/sparse_LR/base_loop.yaml \
#    --model.n_layer 1 \
#    --training.curriculum.loops.start $T \
#    --training.curriculum.loops.end $b \
#    --training.n_loop_window $T \
#    --wandb.name "SparseLR_loop_L1_ends{$b}_T{$T}" \
#    --gpu.n_gpu $n_gpu

# Decision Tree ########################################################################################################
b=70
T=15
#python scripts/train.py --config configs/decision_tree/base_loop.yaml \
#    --model.n_layer 1 \
#    --training.curriculum.loops.start $T \
#    --training.curriculum.loops.end $b \
#    --training.n_loop_window $T \
#    --wandb.name "DT_loop_L1_ends{$b}_T{$T}" \
#    --gpu.n_gpu $n_gpu

# ReLU2NN  #############################################################################################################
b=12
T=5
#python scripts/train.py --config configs/relu_2nn_regression/base_loop.yaml \
#    --model.n_layer 1 \
#    --training.curriculum.loops.start $T \
#    --training.curriculum.loops.end $b \
#    --training.n_loop_window $T \
#    --wandb.name "relu2nn_loop_L1_ends{$b}_T{$T}" \
#    --gpu.n_gpu $n_gpu
