" TẠO 1 CLASS VỚI ĐẦU VÀO LÀ MODEL NAME, --> 1 TENSOR"

from google.colab import drive
drive.mount('/content/drive')

!pip install transformers==4.20.0
!pip install https://github.com/kpu/kenlm/archive/master.zip
!pip install pyctcdecode==0.4.0
!pip install transformers==4.20.0  # Nếu muốn dùng lại cached_path
# hoặc nếu bạn muốn sửa theo version mới hơn:
!pip install huggingface_hub


from huggingface_hub import hf_hub_download
from importlib.machinery import SourceFileLoader
from transformers import Wav2Vec2ProcessorWithLM
import torchaudio
import torch

# Load model name
model_name = "nguyenvulebinh/wav2vec2-base-vi-vlsp2020"

# Tải file model_handling.py từ HuggingFace Hub
model_file = hf_hub_download(repo_id=model_name, filename="model_handling.py")
model_module = SourceFileLoader("model", model_file).load_module()
model = model_module.Wav2Vec2ForCTC.from_pretrained(model_name)

# Load processor with LM decoding
processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_name)

import torch
import torchaudio

# Đường dẫn tới file audio local
audio_file = "/content/drive/MyDrive/workspace/data/audio/audiotest.wav"
waveform, sr = torchaudio.load(audio_file)

# Đảm bảo sample rate là 16kHz như mô hình yêu cầu
if sr != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    waveform = resampler(waveform)
    sr = 16000

# Chuẩn bị input cho model
input_data = processor.feature_extractor(waveform.squeeze(), sampling_rate=sr, return_tensors="pt")

# Dự đoán
with torch.no_grad():
    output = model(**input_data)
    # features = output.last_hidden_state

# ❌ Decode không dùng Language Model (LM)
decoded_nolm = processor.tokenizer.decode(torch.argmax(output.logits, dim=-1)[0])
print("❌ Không dùng LM:", decoded_nolm)

# ✅ Decode có dùng LM
decoded_lm = processor.decode(output.logits[0].cpu().numpy(), beam_width=100).text
print("✅ Dùng LM:", decoded_lm)
# print("feature: ",features)

print("output: ",output.logits.mean(dim=1).shape)




from dotenv import load_dotenv
import os

load_dotenv() 
def audio_features_extractor(path: str)->torch.Tensor:

    # Load model name
    model_name = "nguyenvulebinh/wav2vec2-base-vi-vlsp2020"

    # Tải file model_handling.py từ HuggingFace Hub
    model_file = hf_hub_download(repo_id=model_name, filename="model_handling.py")
    model_module = SourceFileLoader("model", model_file).load_module()
    model = model_module.Wav2Vec2ForCTC.from_pretrained(model_name)

    # Load processor with LM decoding
    processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_name)

    import torch
    import torchaudio

    # Đường dẫn tới file audio local
    audio_file = "/content/drive/MyDrive/workspace/data/audio/audiotest.wav"
    waveform, sr = torchaudio.load(audio_file)

    # Đảm bảo sample rate là 16kHz như mô hình yêu cầu
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        sr = 16000

    # Chuẩn bị input cho model
    input_data = processor.feature_extractor(waveform.squeeze(), sampling_rate=sr, return_tensors="pt")

    # Dự đoán
    with torch.no_grad():
        output = model(**input_data)

    return output.logits.mean(dim=1)