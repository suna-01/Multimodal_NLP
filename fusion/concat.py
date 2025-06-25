# from audio.wav2vec_extractor import Wav2VecExtractor
# from  text.textFeature_extractor import TextFeatureExtractor
# from dotenv import load_dotenv
# import os
# load_dotenv()
# processor_name = os.getenv("PROCESSOR")
# data_path = os.getenv("DATA_PATH")

# processor = Wav2Vec2Processor.from_pretrained(processor_name)
# model = Wav2Vec2Model.from_pretrained(processor_name)
# model.eval()

# Acoustic_features= Wav2VecExtractor(model)

# print("Acoustic_features : ")
# for inputs, labels in dataloader:
#     print(inputs.shape, labels)
#     print("Input shape:", inputs.shape)  # [B, T]
#     features = Acoustic_features.extract(inputs)
#     print("Feature shape:", features.shape)  # [B, T', H]
#     break


# #linguistic feature
# print("linguistic_features : ")

# linguistic=TextFeatureExtractor
# folder =  os.getenv("Data_PATH_text")

# folder_1= os.path.join(folder, "Actor_01")
# extractor = TextFeatureExtractor()
# filenames, features = extractor.extract_from_folder(folder_1)

# print("Số lượng file:", len(filenames))
# print("Shape đặc trưng:", features.shape)  # [N, 768]
# print("Ví dụ:", features[0])



import os
import sys
from dotenv import load_dotenv
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torch.utils.data import TensorDataset, DataLoader

# Thêm đường dẫn thư mục gốc dự án
sys.path.append("D:/code/AI/nlp/multi-model/project/project1")
load_dotenv()
from audio.wav2vec_extractor import Wav2VecExtractor
from text.textFeature_extractor import TextFeatureExtractor
from data.dataloader import RawDataloader
from textData.textDataloader import TextDataLoader

class MultimodalFeatureFusion:
    def __init__(self, audio_model_name, text_model_name, audio_path, text_path, batch_size=8):
        load_dotenv()

        # Acoustic model & processor
        self.processor = Wav2Vec2Processor.from_pretrained(audio_model_name)
        self.audio_model = Wav2Vec2Model.from_pretrained(audio_model_name).eval()
        self.audio_extractor = Wav2VecExtractor(self.audio_model)

        # Text model
        self.text_extractor = TextFeatureExtractor(model_name=text_model_name)

        # Dataloaders
        self.audio_loader = RawDataloader(self.processor, audio_path, batch_size=batch_size, shuffle=False).dataloader
        self.text_loader = TextDataLoader(text_path, batch_size=batch_size).get_dataloader()

    def extract_fused_features(self):
        fused_features = []
        fused_labels = []
        count=0
        for (audio_inputs, labels), text_batch in zip(self.audio_loader, self.text_loader):
            count+=1
            # Acoustic features: [B, T] -> [B, T', H] -> mean pooling -> [B, H]
            acoustic_feat = self.audio_extractor.extract(audio_inputs)  # [B, T', 1024]
            acoustic_feat = acoustic_feat.mean(dim=1)  # [B, 1024]

            # Linguistic features: [B, 768]
            text_feat = self.text_extractor.extract_from_batch(text_batch)

            # Concat: [B, 1024 + 768] = [B, 1792]
            fused = torch.cat([acoustic_feat, text_feat], dim=1)

            fused_features.append(fused)
            fused_labels.append(labels)
            print("========================================================================>")
            if count % 30 == 0 :
                fused_features_step = torch.cat(fused_features, dim=0)
                fused_labels_step = torch.cat(fused_labels, dim=0)
                dataset_step = TensorDataset(fused_features_step, fused_labels_step)
                torch.save(dataset_step, rf"D:\code\AI\nlp\multi-model\project\project1\save1\fused_dataset_step_{count}.pt")


       

        # Gộp lại
        fused_features = torch.cat(fused_features, dim=0)
        fused_labels = torch.cat(fused_labels, dim=0)
        dataset_final=TensorDataset(fused_features, fused_labels)
        torch.save(dataset_step, rf"D:\code\AI\nlp\multi-model\project\project1\save1\fused_dataset_step_final.pt")
        return fused_features, fused_labels
    def get_dataset(self, shuffle=False):
        features, labels = self.extract_fused_features()
        dataset = TensorDataset(features, labels)
        return dataset

    def get_fused_dataloader(self, shuffle=False):
        features, labels = self.extract_fused_features()
        dataset = TensorDataset(features, labels)
        return DataLoader(dataset, batch_size=8, shuffle=shuffle)
