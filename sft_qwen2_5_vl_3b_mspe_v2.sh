export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="llama-factory"
#ps -eo pid,comm | awk '$2~/^python(3)?$/&&$1>10000{print $1}' | xargs --no-run-if-empty kill -9
FORCE_TORCHRUN=1 \
CUDA_VISIBLE_DEVICES=0 \
llamafactory-cli train examples/train_full/qwen2_5vl_3b_full_sft_mspe_v2.yaml
