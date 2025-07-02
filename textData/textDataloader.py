

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
