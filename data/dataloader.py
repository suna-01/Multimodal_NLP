import os
import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from dotenv import load_dotenv
from .dataset import EmotionAudioDataset  # relative import OK nếu dùng như module
from transformers import Wav2Vec2Processor

torchaudio.set_audio_backend("soundfile")
load_dotenv()

class RawDataloader:
    def __init__(self, processor, data_path, batch_size=8, shuffle=False, num_workers=0):
        self.processor = processor
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        # Tạo dataset
        self.dataset = EmotionAudioDataset(self.data_path, self.processor)

        # Tạo dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True,
            collate_fn=self.collate_fn,  # ✅ đã thêm dấu phẩy ở đây
            num_workers=self.num_workers
        )

    def collate_fn(self, batch):
        inputs, labels = zip(*batch)
        inputs_padded = pad_sequence(inputs, batch_first=True)
        labels = torch.tensor(labels)
        return inputs_padded, labels
