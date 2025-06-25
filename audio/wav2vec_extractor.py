import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import os
from dotenv import load_dotenv
from data.dataloader import RawDataloader

# 1. Load biến môi trường
load_dotenv()
processor_name = os.getenv("PROCESSOR")
data_path = os.getenv("DATA_PATH")

# 2. Load processor và model
processor = Wav2Vec2Processor.from_pretrained(processor_name)
model = Wav2Vec2Model.from_pretrained(processor_name)
model.eval()

# 3. Khởi tạo DataLoader từ RawDataloader
dataloader_builder = RawDataloader(processor, data_path)
dataloader = dataloader_builder.dataloader

# 4. Class trích xuất đặc trưng
class Wav2VecExtractor:
    def __init__(self, model):
        self.model = model

    def extract(self, inputs):
        with torch.no_grad():
            features = self.model(inputs).last_hidden_state  # [B, T', H]
        return features

# 5. Sử dụng
extractor = Wav2VecExtractor(model)

# 6. Test một batch
for inputs, labels in dataloader:
    print("Input shape:", inputs.shape)  # [B, T]
    features = extractor.extract(inputs)
    print("Feature shape:", features.shape)  # [B, T', H]
    break
