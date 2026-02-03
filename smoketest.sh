set -ex
rm -rf /tmp/ray/session_* 2>/dev/null || true
export RAY_OBJECT_SPILLING_CONFIG='{"type":"filesystem","params":{"directory_path":"/home/scratch/shrinivr"}}'
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=2,3

# Smoke test 1: baseline (subtree OFF)
# python main.py --env BreakoutNoFrameskip-v4 --case atari --opr train --force \
#   --num_gpus 2 --num_cpus 32 --cpu_actor 8 --gpu_actor 8 \
#   --seed 0 \
#   --p_mcts_num 4 \
#   --use_priority \
#   --use_max_priority \
#   --amp_type 'torch_amp' \
#   --info 'smoke-test' \
#   --object_store_memory 100000000000 \
#   --debug \
#   --subtree_loss_coeff 0.0 \
#   --wandb_project 'MCTS-smoke-test' \
#   --wandb_group_suffix 'baseline_smoketest' 

# Smoke test 2: subtree ON
python main.py --env BreakoutNoFrameskip-v4 --case atari --opr train --force \
  --num_gpus 2 --num_cpus 32 --cpu_actor 8 --gpu_actor 8 \
  --seed 0 \
  --p_mcts_num 4 \
  --use_priority \
  --use_max_priority \
  --amp_type 'torch_amp' \
  --info 'smoke-test' \
  --object_store_memory 100000000000 \
  --debug \
  --subtree_loss_coeff 1.0 \
  --min_visits 5 \
  --wandb_project 'MCTS-smoke-test' \
  --wandb_group_suffix 'subtree-on' \
  --wandb_offline
