import numpy as np

## Tarefa 1: Implementando a Máscara Causal
# objetivos : 1, 2 e 3
def create_causal_mask(seq_len):
    # Criar uma máscara causal utilizando NumPy
    mask = np.zeros((seq_len, seq_len))

    indices_superior = np.triu_indices(seq_len, k=1)

    mask[indices_superior] = -np.inf

    return mask

#print(f"Máscara Causal (Tamanho {seq_len}x{seq_len}):")
#print(mask)

def Softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


seq_len = 5
dimensao_vetores = 4

np.random.seed(42) # Para você ver os mesmos resultados que eu
Q = np.random.randn(seq_len, dimensao_vetores)
K = np.random.randn(seq_len, dimensao_vetores)

scores = np.dot(Q, K.T)

mask = create_causal_mask(seq_len)

masked_scores = scores + mask


probabilidades = Softmax(masked_scores)

print(f"Matriz de Scores Originais (Sem Máscara):\n{np.round(scores, 2)}")

print(f"\nMatriz de Scores Mascarados (Com -inf):\n{np.round(masked_scores,2)}")

print(f"\nProbabilidades Finais (Softmax):")
np.set_printoptions(suppress=True, precision=4)
print(np.round(probabilidades,2))

## Tarefa 2: A Ponte Encoder-Decoder (Cross-Attention)

