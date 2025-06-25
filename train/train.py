import os
import sys
sys.path.append("D:/code/AI/nlp/multi-model/project/project1")
from dotenv import load_dotenv
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torch.utils.data import TensorDataset, DataLoader
from fusion.concat import MultimodalFeatureFusion
# Thêm đường dẫn thư mục gốc dự án

load_dotenv()
from audio.wav2vec_extractor import Wav2VecExtractor
from text.textFeature_extractor import TextFeatureExtractor
from data.dataloader import RawDataloader
from textData.textDataloader import TextDataLoader

audio_model = os.getenv("PROCESSOR")
text_model = "microsoft/deberta-v3-base"
audio_path = os.getenv("DATA_PATH")
text_path = os.getenv("Data_PATH_text")
print(audio_model)
print(text_model)

print(audio_path)

print(text_path)



fusion = MultimodalFeatureFusion(audio_model, text_model, audio_path, text_path)
dataloader = fusion.get_fused_dataloader()

for x, y in dataloader:
    print("✅ Fused shape:", x.shape)  # [B, 1792]
    print("🎯 Labels:", y)
    # break