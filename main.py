import torch # 

def create_causal_mask(seq_len = 5):
    # Criar uma máscara causal (triangular superior) para seq_len 
    mask = touch.triu(touch.ones(seq_len, seq_len), diagonal=1)
