######
######## generate dataset
#python scripts/gen_openml_data.py
######
#### run experiments
#for test_id in 0 1 2 3 4 5 6 7 8 9
#do
#  python scripts/train_openml.py --config configs/openml/base.yaml \
#      --wandb.name "openml_baseline_test_id_{$test_id}" \
#      --gpu.n_gpu 1 \
#      --training.batch_size 40 \
#      --training.train_steps 100000 \
#      --training.test_idx $test_id
#done
#
#
#T=15
#b=30
#for test_id in 0 1 2 3 4 5 6 7 8 9
#do
#  python scripts/train_openml.py --config configs/openml/base_loop.yaml \
#      --wandb.name "openml_loop_test_id_{$test_id}" \
#      --model.n_layer 1 \
#      --training.curriculum.loops.start $T \
#      --training.curriculum.loops.end $b \
#      --training.n_loop_window $T \
#      --gpu.n_gpu 0 \
#      --training.train_steps 100000 \
#      --training.batch_size 40 \
#      --training.test_idx $test_id
#done