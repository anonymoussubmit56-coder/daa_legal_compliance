import pandas as pd
from evaluate import load
from bert_score import score
import math
from collections import Counter

# -------------------------------
# Función para entropía léxica
# -------------------------------
def lexical_entropy(text):
    words = text.split()
    total_words = len(words)
    if total_words == 0:
        return 0.0
    word_counts = Counter(words)
    probs = [count / total_words for count in word_counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return entropy

# -------------------------------
# Cargar métricas
# -------------------------------
rouge = load("rouge")

# -------------------------------
# Cargar CSV y procesar
# -------------------------------
df = pd.read_csv("prompt_novice.csv")

# Filtrar solo filas con referencia válida
df = df[df['reference'].notnull() & (df['reference'].str.strip() != "")]

# Inicializar listas para almacenar resultados
rouge_scores = []
bert_f1_scores = []
entropies = []
lengths = []

# Procesar cada fila
for i, row in df.iterrows():
    pred = row['response']
    ref = row['reference']

    # ROUGE
    rouge_result = rouge.compute(predictions=[pred], references=[ref])
    rouge_l = rouge_result['rougeL']
    rouge_scores.append(rouge_l)

    # BERTScore
    _, _, f1 = score([pred], [ref], lang="en", verbose=False)
    bert_f1_scores.append(f1[0].item())

    # Entropía
    entropies.append(lexical_entropy(pred))

    # Longitud
    lengths.append(len(pred.split()))

# Agregar columnas al DataFrame original
df['rougeL'] = rouge_scores
df['bert_score_f1'] = bert_f1_scores
df['entropy'] = entropies
df['response_length'] = lengths

# Guardar resultados
df.to_csv("evaluated_responses.csv", index=False)
print("✅ Métricas calculadas y guardadas en 'evaluated_responses.csv'")
