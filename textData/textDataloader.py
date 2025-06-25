# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer
# from .textDataset import TextDataset  # hoặc điều chỉnh import theo cấu trúc bạn
# import torch

# class TextDataLoader:
#     def __init__(self, text_folder, model_name="microsoft/deberta-v3-base", batch_size=8):
#         self.dataset = TextDataset(text_folder)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.batch_size = batch_size

#     def collate_fn(self, batch_texts):
#         inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
#         return inputs

#     def get_dataloader(self):
#         return DataLoader(
#             self.dataset,
#             batch_size=self.batch_size,
#             shuffle=False,  # ❗ không được shuffle để giữ thứ tự khớp audio
#             collate_fn=self.collate_fn
#         )
# textData/textDataloader.py

from torch.utils.data import DataLoader
from textData.textDataset import TextDataset

class TextDataLoader:
    def __init__(self, text_folder, batch_size=8):
        self.dataset = TextDataset(text_folder)
        self.batch_size = batch_size

    def get_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False  # giữ nguyên thứ tự file
        )
