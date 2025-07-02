import os
import torch
import torchaudio
# from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# Cấu hình
AUDIO_ROOT = os.getenv("Data_PATH")
TEXT_ROOT = os.getenv("Data_PATH_text")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

model.eval()

# Set backend
torchaudio.set_audio_backend("soundfile")

def transcribe(wav_path):
    waveform, sr = torchaudio.load(wav_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
        sr = 16000

    with torch.no_grad():
        inputs = processor(waveform.squeeze(), sampling_rate=sr, return_tensors="pt")
        logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

    return transcription
# Duyệt thư mục không xáo trộn
for root, dirs, files in os.walk(AUDIO_ROOT):
    dirs.sort()   # Đảm bảo thứ tự thư mục
    files.sort()  # Đảm bảo thứ tự file

    for file in files:
        if file.endswith(".wav"):
            wav_path = os.path.join(root, file)
            rel_path = os.path.relpath(wav_path, AUDIO_ROOT)
            txt_path = os.path.join(TEXT_ROOT, os.path.splitext(rel_path)[0] + ".txt")

            os.makedirs(os.path.dirname(txt_path), exist_ok=True)

            try:
                text = transcribe(wav_path)
            except Exception as e:
                print(f" Lỗi xử lý {wav_path}: {e}")
                continue

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text.strip())
            print(f" {txt_path}")
           
