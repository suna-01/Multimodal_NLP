# textData/textDataset.py

import os
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text_folder):
        self.files = []
        for root, _, files in os.walk(text_folder):
            for f in sorted(files):
                if f.endswith(".txt"):
                    self.files.append(os.path.join(root, f))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        return text
