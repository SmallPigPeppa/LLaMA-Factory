from transformers import AutoConfig, AutoImageProcessor
import os

# 1. 注册自定义 ImageProcessor
from custom_models.qwen2_5_vl_3b_mspe.A_image_processor import Qwen2VLImageProcessor
AutoConfig.register("Qwen2VLImageProcessorMSPE", Qwen2VLImageProcessor)


# 3. 用 from_pretrained 自动加载
processor = AutoImageProcessor.from_pretrained("custom_models/qwen2_5_vl_3b_mspe")

print("加载的 processor 实例：", processor)
print("类型：", type(processor))
