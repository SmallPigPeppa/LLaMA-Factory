export WANDB_PROJECT="llama-factory"
FORCE_TORCHRUN=1 \
CUDA_VISIBLE_DEVICES=0 \
llamafactory-cli train examples/train_full/qwen2_5vl_3b_full_sft.yaml
