import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv("e_p.csv")

# clean
df = df.dropna(subset=["response", "reference"])

# =========================================================
# ROUGE-L
# =========================================================
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def compute_rouge(row):
    try:
        return scorer.score(row["reference"], row["response"])["rougeL"].fmeasure
    except:
        return np.nan

df["rougeL"] = df.apply(compute_rouge, axis=1)

# =========================================================
# BERTScore (lightweight mode)
# =========================================================
P, R, F1 = bert_score(
    df["response"].tolist(),
    df["reference"].tolist(),
    lang="en",
    verbose=False
)

df["bert_f1"] = F1.numpy()

# =========================================================
# LENGTH (controllability proxy)
# =========================================================
df["response_length"] = df["response"].apply(lambda x: len(str(x).split()))

# =========================================================
# SPLIT NOVICE / EXPERT
# =========================================================
novice = df[df["alpha"] == 0.1]
expert = df[df["alpha"] == 0.9]

# =========================================================
# TABLE 1: MAIN RQ2 COMPARISON
# =========================================================
table_main = pd.DataFrame({
    "Setting": ["Novice (α=0.1)", "Expert (α=0.9)"],
    "ROUGE-L": [novice["rougeL"].mean(), expert["rougeL"].mean()],
    "BERTScore": [novice["bert_f1"].mean(), expert["bert_f1"].mean()],
    "Length": [novice["response_length"].mean(), expert["response_length"].mean()],
})

print("\n=== TABLE 1 (RQ2 MAIN) ===")
print(table_main)

# =========================================================
# TABLE 2: SEMANTIC GAP
# =========================================================
table_semantic = pd.DataFrame({
    "Setting": ["Novice", "Expert"],
    "ROUGE-L": [novice["rougeL"].mean(), expert["rougeL"].mean()],
    "BERTScore": [novice["bert_f1"].mean(), expert["bert_f1"].mean()],
})

print("\n=== TABLE 2 (SEMANTIC) ===")
print(table_semantic)

# =========================================================
# TABLE 3: COMPLEXITY / CONTROLLABILITY
# =========================================================
table_complexity = pd.DataFrame({
    "Setting": ["Novice", "Expert"],
    "Length": [novice["response_length"].mean(), expert["response_length"].mean()],
})

print("\n=== TABLE 3 (COMPLEXITY) ===")
print(table_complexity)

# =========================================================
# TABLE 4: STABILITY (VERY IMPORTANT FOR EMNLP)
# =========================================================
table_stability = df.groupby("question").agg({
    "rougeL": "std",
    "bert_f1": "std",
    "response_length": "std"
}).reset_index()

print("\n=== TABLE 4 (STABILITY PER QUESTION) ===")
print(table_stability.head())

# =========================================================
# SAVE
# =========================================================
df.to_csv("rq2_processed.csv", index=False)
table_main.to_csv("rq2_table_main.csv", index=False)
table_semantic.to_csv("rq2_table_semantic.csv", index=False)
table_complexity.to_csv("rq2_table_complexity.csv", index=False)
table_stability.to_csv("rq2_table_stability.csv", index=False)