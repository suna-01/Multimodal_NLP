import os
import torch
import torchaudio
from torch.utils.data import Dataset
from dotenv import load_dotenv
torchaudio.set_audio_backend("soundfile")
load_dotenv()
# Ánh xạ ID trong tên file → tên cảm xúc
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Ánh xạ tên cảm xúc → số (label)
label2id = {v: i for i, v in enumerate(emotion_map.values())}


class EmotionAudioDataset(Dataset):
    def __init__(self, data_path, processor, max_len=5.0):
        """
        data_path: thư mục chứa 24 folder audio (RAVDESS)
        processor: Wav2Vec2Processor (hoặc tương đương)
        max_len: giới hạn độ dài audio (giây)
        """
        self.data_path = data_path
        self.processor = processor
        self.max_len = max_len
        self.data = []

        # Duyệt các file âm thanh .wav
        for folder in os.listdir(data_path):
            folder_path = os.path.join(data_path, folder)
            if not os.path.isdir(folder_path):
                continue

            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    parts = file.split("-")
                    if len(parts) < 3:
                        continue  # Tránh lỗi nếu tên không đúng định dạng

                    emotion_id = parts[2]
                    if emotion_id in emotion_map:
                        label_name = emotion_map[emotion_id]
                        label = label2id[label_name]
                        file_path = os.path.join(folder_path, file)
                        self.data.append((file_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        waveform, sr = torchaudio.load(path)

        # Chuyển stereo → mono nếu cần
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample về 16kHz nếu cần
        if sr != 16000:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resample(waveform)

        # Cắt nếu dài quá
        max_samples = int(self.max_len * 16000)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

        # Chuẩn hóa và trích đặc trưng
        inputs = self.processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)

        return inputs.input_values.squeeze(0), torch.tensor(label)


