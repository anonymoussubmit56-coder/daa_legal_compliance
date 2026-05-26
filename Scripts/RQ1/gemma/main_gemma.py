#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from datetime import datetime


# === CONFIGURATION ===
FILENAME = "context.txt"
SIMILARITY_THRESHOLD = 0.4 # Steering strength

# === LOGGING FUNCTION ===

def write_log(message, log_file="log.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}\n"
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(full_message)

# === LOAD CONTEXT ===
def read_file_lines(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return [line.strip().lower() for line in file.readlines()]
    
def load_questions(filepath):
    questions = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                questions.append(line)
    return questions

def initialize_models():
    """Load models once and return them."""
    write_log("Loading models...")
    
    # === DEVICE SETUP (Apple Silicon) ===
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. EMBEDDING MODEL
    embedding_model = SentenceTransformer("all-mpnet-base-v2")
    
    # 2. PHI-2 (LLM)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")
    model.to(device)
    model.eval()
    
    # 3. Projection Layer (Se inicializa una vez)
    target_embedding_dim = model.get_input_embeddings().embedding_dim
    projection_layer = torch.nn.Linear(768, target_embedding_dim).to(device)
    
    return embedding_model, tokenizer, model, projection_layer, device


def run_batch(filepath, alpha_values=[0.1, 0.5, 0.9]):
    questions = load_questions(filepath)
    contexts = read_file_lines(FILENAME)
    
    embedding_model, tokenizer, model, projection_layer, device = initialize_models()
    write_log("Pre-calculating context embeddings...")
    context_embeddings = embedding_model.encode(contexts, convert_to_tensor=True)
    write_log("Context embeddings pre-calculated.")
    
    for prompt in questions:
        if prompt == "":
            continue
        for alpha in alpha_values:
            write_log("-------------------------------------------------------------------------------------------")
            write_log(f"\nRunning: '{prompt}' with ALPHA={alpha}")
            print(f"Running: '{prompt}' with ALPHA={alpha}")
            main(prompt, ALPHA=alpha, contexts=contexts, 
                 embedding_model=embedding_model, 
                 context_embeddings=context_embeddings,
                 tokenizer=tokenizer, 
                 model=model, 
                 projection_layer=projection_layer, 
                 device=device)


def main(prompt, ALPHA, contexts, embedding_model, context_embeddings, tokenizer, model, projection_layer, device):
    prompt = prompt.lower()    # Ensure prompt is lowercase
    write_log(f"Similarity threshold set to {SIMILARITY_THRESHOLD}")
    write_log(f"Steering strength (ALPHA) set to {ALPHA}")

    # === 1. COMPUTE PROMPT EMBEDDING ===
    prompt_embedding = embedding_model.encode(prompt, convert_to_tensor=True).to(device)

    # === 2. FIND RELEVANT CONTEXT (Vectorizado) ===
    # Calcula la similitud coseno del prompt contra TODOS los embeddings del contexto
    # context_embeddings ya está en device (si se inicializó correctamente)
    scores = util.cos_sim(prompt_embedding, context_embeddings)[0]
    
    # Filtra los embeddings relevantes que cumplen con el umbral (SIMILARITY_THRESHOLD)
    relevant_embeddings = context_embeddings[scores >= SIMILARITY_THRESHOLD]

    if relevant_embeddings.numel() == 0:
        write_log("No relevant context found.")
        return

    # === 3. COMPUTE STEERING VECTOR ===
    # El vector de dirección es el promedio de los embeddings relevantes
    steering_vector = relevant_embeddings.mean(dim=0)
    
    # Verifica la validez del vector de dirección resultante
    similarity = util.cos_sim(prompt_embedding, steering_vector).item()
    if similarity < SIMILARITY_THRESHOLD:
        write_log("Context is NOT valid — skipping generation.")
        return

    write_log(f"Context is valid. Similarity score: {similarity}")

    # === 4. PREPARE MODEL INPUTS ===
    
    # Tokeniza el prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"] 
    
    # Obtiene los embeddings de entrada
    embedding_layer = model.get_input_embeddings()
    input_embeddings = embedding_layer(input_ids)

    # Proyecta el steering vector (768 → 2048)
    steering_vector_projected = projection_layer(steering_vector.to(device))

    # === 5. APPLY STEERING TO FIRST TOKEN ===
    # Aplica la fuerza de steering (ALPHA) al primer token del prompt
    input_embeddings[0, 0] = (1 - ALPHA) * input_embeddings[0, 0] + ALPHA * steering_vector_projected

    # Crea la máscara de atención
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)

    # === 6. GENERATE OUTPUT ===
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            max_new_tokens=100,
            num_beams=4,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # === 7. DECODE AND LOG RESULT ===
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    write_log("Generated response: \n" + response)

if __name__ == "__main__":
    try:
        run_batch('questions.txt')
    except Exception as e:
        write_log(f"Error{str(e)}")
    finally:
        write_log("\n------------------------------------------------------------------------------------------\n")

