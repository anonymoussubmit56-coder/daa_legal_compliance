#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

# === CONFIGURATION ===
FILENAME = "context.txt"
# SIMILARITY_THRESHOLD ya no se usa en este script, pero se mantiene la estructura.

# === LOGGING FUNCTION ===
def write_log(message, log_file="log.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}\n"
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(full_message)

# === LOAD CONTEXT & QUESTIONS ===
def read_file_lines(filename):
    # Esta función ya no es estrictamente necesaria si no se usa el contexto
    # pero la mantenemos para compatibilidad si se reintroduce la lógica de contexto.
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

# -------------------------------------------------------------------------

# === NUEVA FUNCIÓN: INICIALIZACIÓN ÚNICA DEL LLM ===
def initialize_llm():
    """Define el dispositivo y carga el modelo LLM una única vez."""
    write_log("Initializing LLM and device...")
    
    # === DEVICE SETUP ===
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # === LOAD PHI-2 ===
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", 
        torch_dtype=torch.float32, 
        trust_remote_code=True
    ).to(device)
    
    # Se añade model.eval() para optimizar inferencia
    model.eval() 
    
    write_log(f"Model loaded successfully on device: {device}")
    return model, tokenizer, device

# -------------------------------------------------------------------------

# === FUNCIÓN run_batch MODIFICADA ===
def run_batch(filepath, alpha_values=[0.1, 0.5, 0.9]):
    questions = load_questions(filepath)
    
    # >>> OPTIMIZACIÓN CLAVE: Carga Única del Modelo <<<
    model, tokenizer, device = initialize_llm()
    
    for prompt in questions:
        if not prompt:
            continue
            
        write_log("-------------------------------------------------------------------------------------------")
        write_log(f"\nRunning: '{prompt}'")
        print(f"Running: '{prompt}'")
        
        # Pasa los objetos pre-cargados a main
        main(prompt, model=model, tokenizer=tokenizer, device=device) 

# -------------------------------------------------------------------------

# === FUNCIÓN main MODIFICADA Y OPTIMIZADA ===
def main(prompt, model, tokenizer, device):
    prompt = "Explain this for a expert audience "+prompt.lower()  # Ensure prompt is lowercase
    
    # Tokeniza el prompt y lo mueve al dispositivo
    # Se quita 'return_attention_mask=False' para usar la función `model.generate` de forma más estándar.
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # input_ids = inputs["input_ids"] # Ya no es necesario obtener `input_ids` por separado

    # Genera la salida.
    with torch.no_grad(): # Añadido para ahorrar memoria y acelerar la inferencia
        outputs = model.generate(
            **inputs, # Pasa `input_ids` y `attention_mask` automáticamente
            pad_token_id=tokenizer.eos_token_id,
            max_length=200
        )
        
    text = tokenizer.batch_decode(outputs)[0]
    print(text)
    write_log("Generated response: \n" + text)

# -------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run_batch('questions.txt')
    except Exception as e:
        write_log(f"Error: {str(e)}")
    finally:
        write_log("\n------------------------------------------------------------------------------------------\n")
