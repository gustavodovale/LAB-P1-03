import numpy as np

## Tarefa 1: Implementando a Máscara Causal

def create_causal_mask(seq_len):
    # Criar uma máscara causal utilizando NumPy
    mask = np.zeros((seq_len, seq_len))
    # Seleciona índices acima da diagonal (k=1) para aplicar o veto
    indices_superior = np.triu_indices(seq_len, k=1)

    mask[indices_superior] = -np.inf

    return mask

def Softmax(x):
    # Subtração do max evita overflow exponencial (estabilidade numérica)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


seq_len = 5
dimensao_vetores = 4

np.random.seed(42) # Para você ver os mesmos resultados que eu
Q = np.random.randn(seq_len, dimensao_vetores)
K = np.random.randn(seq_len, dimensao_vetores)

# Produto escalar: mede a afinidade entre cada par de tokens
scores = np.dot(Q, K.T)

mask = create_causal_mask(seq_len)

# Aplica a máscara: scores do futuro tornam-se -inf
masked_scores = scores + mask

# Softmax converte -inf em 0.0 absoluto (proibição de olhar adiante)
probabilidades = Softmax(masked_scores)

print(f"Matriz de Scores Originais (Sem Máscara):\n{np.round(scores, 2)}")

print(f"\nMatriz de Scores Mascarados (Com -inf):\n{np.round(masked_scores,2)}")

print(f"\nProbabilidades Finais (Softmax):")
np.set_printoptions(suppress=True, precision=4)
print(np.round(probabilidades,2))



## Tarefa 2: A Ponte Encoder-Decoder (Cross-Attention)

def Cruzamento_atensão(encoder_out, decoder_state):
    d_model = encoder_out.shape[-1]
    print(d_model)
    d_k = d_model # dimensão das chaves

    # Matrizes de Projeção: transformam os estados em espaços de busca (Q, K, V)
    Q_peso = np.random.randn(d_model, d_model)
    K_peso = np.random.randn(d_model, d_model)
    V_peso = np.random.randn(d_model, d_model)

    # Projeções lineares (Transformações de espaço)
    Q = np.dot(decoder_state, Q_peso) 
    K = np.dot(encoder_out, K_peso)   
    V = np.dot(encoder_out, V_peso)   

    # Scaled Dot-Product: calcula relevância do Decoder sobre a memória do Encoder
    scores = np.dot(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    # Gera o mapa de alinhamento (quais palavras da origem importam para o alvo)
    atensao_pesos = Softmax(scores)
    
    # Combinação Convexa: agrega os Valores (V) baseada nos pesos de atenção
    output = np.dot(atensao_pesos, V)
    
    return output, atensao_pesos


# Simulação Tarefa 2
seq_len_frances=10
seq_len_ingles=4

# Saída do Encoder
encoder_output = np.random.randn(1, seq_len_frances, 512)

# Estado do Decoder
decoder_state = np.random.randn(1, seq_len_ingles, 512)

context_vector, pesos = Cruzamento_atensão(encoder_output, decoder_state)

print(f"Dimensão da saída do Cross-Attention: {context_vector.shape}")
print(f"Dimensão da matriz de pesos (Alinhamento): {pesos.shape}")


## Tarefa 3: Simulando o Loop de Inferência Auto-Regressivo

VOCABULARIO = ["<SOS>", "O", "rato", "roeu", "a", "roupa", "do", "rei", "de", "Roma", "<EOS>"]
V_SIZE = 10000 # Tamanho do vocabulário total (ex: 10k)
TOKEN_PARA_ID = {word: i for i, word in enumerate(VOCABULARIO)}
ID_PARA_TOKEN = {i: word for i, word in enumerate(VOCABULARIO)}






