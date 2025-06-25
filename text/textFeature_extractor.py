import os
import torch
from transformers import DebertaV2Tokenizer, DebertaV2Model
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# class TextFeatureExtractor:
#     def __init__(self, model_name="microsoft/deberta-v3-base", device=None):
#         self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
#         self.model = DebertaV2Model.from_pretrained(model_name)
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
#         self.model.eval()

#     def extract_from_texts(self, texts):
#         """
#         texts: List[str] — các câu văn
#         return: Tensor [B, hidden_dim] — đặc trưng đầu ra (mean pooling)
#         """
#         with torch.no_grad():
#             inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
#             outputs = self.model(**inputs)
#             hidden_states = outputs.last_hidden_state  # [B, T, H]
#             mask = inputs['attention_mask'].unsqueeze(-1)  # [B, T, 1]
#             masked_hidden = hidden_states * mask
#             mean_embeddings = masked_hidden.sum(dim=1) / mask.sum(dim=1)
#         return mean_embeddings.cpu()  # [B, H]

#     def extract_from_folder(self, text_folder, max_files=None):
#         """
#         text_folder: thư mục chứa các file .txt
#         return: (List[str], Tensor [N, H]) — tên file và embedding
#         """
#         embeddings = []
#         filenames = []
#         all_files = []

#         for root, _, files in os.walk(text_folder):
#             for f in sorted(files):
#                 if f.endswith(".txt"):
#                     full_path = os.path.join(root, f)
#                     all_files.append(full_path)
#         if max_files:
#             all_files = all_files[:max_files]

#         texts = []
#         for file in tqdm(all_files, desc="Reading and extracting text"):
#             with open(file, 'r', encoding='utf-8') as f:
#                 text = f.read().strip()
#                 if text:
#                     texts.append(text)
#                     filenames.append(file)
#                 else:
#                     print(f"⚠️ Empty: {file}")

#         # Batch processing (nếu quá nhiều file)
#         BATCH_SIZE = 32
#         all_features = []
#         for i in range(0, len(texts), BATCH_SIZE):
#             batch = texts[i:i + BATCH_SIZE]
#             features = self.extract_from_texts(batch)  # [B, H]
#             all_features.append(features)

#         return filenames, torch.cat(all_features, dim=0)  # [N, H]
    
# folder =  os.getenv("Data_PATH_text")
# folder_1= os.path.join(folder, "Actor_01")
# extractor = TextFeatureExtractor()
# filenames, features = extractor.extract_from_folder(folder_1)

# print("Số lượng file:", len(filenames))
# print("Shape đặc trưng:", features.shape)  # [N, 768]
# print("Ví dụ:", features[0])







# text/textFeature_extractor.py

import torch
from transformers import AutoTokenizer, AutoModel

class TextFeatureExtractor:
    def __init__(self, model_name="microsoft/deberta-v3-base", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def extract_from_batch(self, batch_texts):
        """
        batch_texts: List[str] — mỗi câu trong batch
        return: Tensor [B, hidden_dim] — đặc trưng đầu ra (mean pooling)
        """
        with torch.no_grad():
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # [B, T, H]
            mask = inputs['attention_mask'].unsqueeze(-1)  # [B, T, 1]
            masked_hidden = hidden_states * mask
            mean_embeddings = masked_hidden.sum(dim=1) / mask.sum(dim=1)
        return mean_embeddings.cpu()  # [B, H]
