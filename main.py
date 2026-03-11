import numpy as np

## Tarefa 1: Implementando a Máscara Causal

def create_causal_mask(seq_len):
    # Criar uma máscara causal utilizando NumPy
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) 
    return mask

seq_len = 5
mask = create_causal_mask(seq_len)

print("Máscara Causal (NumPy):")
print(mask)
