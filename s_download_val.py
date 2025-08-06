from tqdm import  tqdm
from datasets import load_dataset
data = load_dataset("lmms-lab/LLaVA-NeXT-Data", cache_dir='/mnt/bn/liuwenzhuo-hl-data/hf_cache', split="train")
for da in tqdm(data):
    error_ids = []
    for i, da in enumerate(tqdm(data)):
        # 假设 da 结构大致是 {'messages': [...] , ...}
        for msg in da['messages']:
            content = msg.get('content', '')
            image_count = content.count('<image>')
            # 通常图片都在 images 字段，或类似 images/paths 字段
            images = da.get('images', [])  # 具体字段名请根据实际情况调整
            if isinstance(images, str):
                images = [images] if images else []
            if image_count != len(images):
                print(f"问题样本 index={i}：<image>数量={image_count}, 图片数量={len(images)}")
                error_ids.append(i)
