from dotenv import load_dotenv
import os
import torch
load_dotenv() 

def audio_feature_extractor(text:str)->torch.Tensor:
    '''
    Đầu vào là 1 chuỗi text từ audio và trả về 1 tensor lưu trữ thông tin về quan hệ giữ các từ
    '''

    
    phobert = AutoModel.from_pretrained("vinai/phobert-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    line = text
    input_ids = torch.tensor([tokenizer.encode(line)])

    with torch.no_grad():
        features = phobert(input_ids)  # Models outputs are now tuples


    # 4. Lấy last hidden state (embedding cho từng token)
    last_hidden_state = features.last_hidden_state  # hoặc outputs[0] cũng được
    print("Shape của output:", last_hidden_state.mean(dim=1).shape)  # (batch_size, seq_len, hidden_size)
    print("feature_tensor: ",features)
    return last_hidden_state.mean(dim=1)