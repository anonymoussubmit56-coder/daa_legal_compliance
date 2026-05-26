#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from datetime import datetime

# === CONFIGURATION ===
FILENAME = "context.txt"
SIMILARITY_THRESHOLD = 0.4  # Steering strength

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

# === INITIALIZE MODELS ===
def initialize_models():
    write_log("Loading models...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_log(f"Using device: {device}")

    # Embedding model (on GPU if available)
    embedding_model = SentenceTransformer(
        "all-mpnet-base-v2",
        device=device
    )

    # LLM 
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-base",
        dtype=torch.float32,
        device_map="auto"
    )
    model.eval()

    # Projection layer
    target_embedding_dim = model.get_input_embeddings().embedding_dim
    projection_layer = torch.nn.Linear(768, target_embedding_dim).to(device)

    return embedding_model, tokenizer, model, projection_layer, device

# === RUN BATCH ===
def run_batch(filepath, alpha_values=[0.1, 0.5, 0.9]):
    questions = load_questions(filepath)
    contexts = read_file_lines(FILENAME)

    embedding_model, tokenizer, model, projection_layer, device = initialize_models()

    write_log("Pre-calculating context embeddings...")
    context_embeddings = embedding_model.encode(
        contexts,
        convert_to_tensor=True
    )
    write_log("Context embeddings pre-calculated.")

    for prompt in questions:
        if prompt == "":
            continue
        for alpha in alpha_values:
            write_log("-------------------------------------------------------------------------------------------")
            write_log(f"Running: '{prompt}' with ALPHA={alpha}")
            print(f"Running: '{prompt}' with ALPHA={alpha}")

            main(
                prompt,
                ALPHA=alpha,
                contexts=contexts,
                embedding_model=embedding_model,
                context_embeddings=context_embeddings,
                tokenizer=tokenizer,
                model=model,
                projection_layer=projection_layer,
                device=device
            )

# === MAIN LOGIC ===
def main(prompt, ALPHA, contexts, embedding_model, context_embeddings,
         tokenizer, model, projection_layer, device):

    prompt = prompt.lower()  
    write_log(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    write_log(f"Steering strength (ALPHA): {ALPHA}")

    # 1. Prompt embedding
    prompt_embedding = embedding_model.encode(
        prompt,
        convert_to_tensor=True
    )

    # 2. Cosine similarity
    scores = util.cos_sim(prompt_embedding, context_embeddings)[0]
    relevant_embeddings = context_embeddings[scores >= SIMILARITY_THRESHOLD]

    if relevant_embeddings.numel() == 0:
        write_log("No relevant context found.")
        return

    # 3. Steering vector
    steering_vector = relevant_embeddings.mean(dim=0)
    similarity = util.cos_sim(prompt_embedding, steering_vector).item()

    if similarity < SIMILARITY_THRESHOLD:
        write_log("Context is NOT valid — skipping generation.")
        return

    write_log(f"Context valid. Similarity: {similarity}")

    # 4. Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    embedding_layer = model.get_input_embeddings()
    input_embeddings = embedding_layer(input_ids)

    # 5. Apply steering
    steering_vector_projected = projection_layer(steering_vector.to(device))
    input_embeddings[0, 0] = (
        (1 - ALPHA) * input_embeddings[0, 0]
        + ALPHA * steering_vector_projected
    )

    attention_mask = torch.ones(
        input_ids.shape,
        dtype=torch.long,
        device=device
    )

    # 6. Generate
    write_log("Starting generation...")
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )
    write_log("Generation finished.")

    # 7. Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    write_log("Generated response:\n" + response)

# === ENTRY POINT ===
if __name__ == "__main__":
    try:
        run_batch("questions.txt")
    except Exception as e:
        write_log(f"ERROR: {str(e)}")
    finally:
        write_log("\n------------------------------------------------------------------------------------------\n")

