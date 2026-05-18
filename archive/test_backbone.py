from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
import numpy as np
import torch


config = AutoConfig.from_pretrained("OpenGVLab/VideoMAEv2-Base", trust_remote_code=True)
processor = VideoMAEImageProcessor.from_pretrained("OpenGVLab/VideoMAEv2-Base")
model = AutoModel.from_pretrained('OpenGVLab/VideoMAEv2-Base', config=config, trust_remote_code=True)


video = list(np.random.rand(16, 3, 224, 224))




# B, T, C, H, W -> B, C, T, H, W
inputs = processor(video, return_tensors="pt")
inputs['pixel_values'] = inputs['pixel_values'].permute(0, 2, 1, 3, 4)

with torch.no_grad():
  outputs = model(**inputs)
