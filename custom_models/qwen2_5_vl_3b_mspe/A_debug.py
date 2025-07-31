from transformers import AutoImageProcessor

from custom_models.qwen2_5_vl_3b_mspe.A_image_processor import Qwen2VLImageProcessor
AutoImageProcessor.register("Qwen2VLImageProcessorMSPE", Qwen2VLImageProcessor)


processor = AutoImageProcessor.from_pretrained("custom_models/qwen2_5_vl_3b_mspe")
