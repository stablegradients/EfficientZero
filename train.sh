set -ex
rm -rf /tmp/ray/session_* 2>/dev/null || true
export RAY_OBJECT_SPILLING_CONFIG='{"type":"filesystem","params":{"directory_path":"/home/scratch/shrinivr"}}'
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=2,3

python main.py --env BreakoutNoFrameskip-v4 --case atari --opr train --force \
  --num_gpus 2 --num_cpus 32 --cpu_actor 8 --gpu_actor 8 \
  --seed 0 \
  --p_mcts_num 4 \
  --use_priority \
  --use_max_priority \
  --amp_type 'torch_amp' \
  --info 'EfficientZero-V1'\
  --object_store_memory 100000000000 \
  --debug
