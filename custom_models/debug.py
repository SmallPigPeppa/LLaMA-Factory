from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "/home/tiger/LLaMA-Factory/custom_models/qwen2_5_vl_mspe/",
    trust_remote_code=True
)
print(type(model))