import os
import torch
import json
import gc
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, PeftModel

# Configurar entorno para Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Liberar memoria
def clear_memory():
    gc.collect()
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    print("✓ Memoria liberada")

# Verificar si estamos en Apple Silicon
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device = get_device()
print(f"Usando dispositivo: {device}")

# Configuración general
MODEL_ID = "microsoft/phi-2"
OUTPUT_DIR = "./lora_phi2_output"
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_R = 16
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 2e-4
NUM_EPOCHS = 5
BATCH_SIZE = 1

# Cargar el dataset desde archivo JSON
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Asegurarse de que el formato sea correcto
    processed_data = []
    for item in data:
        if isinstance(item, dict) and "instruction" in item and "output" in item:
            if "input" not in item:
                item["input"] = ""
            processed_data.append(item)
    
    dataset_dict = {
        "instruction": [item["instruction"] for item in processed_data],
        "input": [item["input"] for item in processed_data],
        "output": [item["output"] for item in processed_data]
    }
    
    return Dataset.from_dict(dataset_dict)

# Función para formatear las entradas según el modelo utilizado
def format_prompt(example):
    # Formato para phi-2
    if example["input"]:
        text = f"Instrucción: {example['instruction']}\nEntrada: {example['input']}\nRespuesta: {example['output']}"
    else:
        text = f"Instrucción: {example['instruction']}\nRespuesta: {example['output']}"
    
    return text

# Función para tokenizar el dataset
def tokenize_dataset(example, tokenizer):
    prompt = format_prompt(example)
    
    # Tokenizar con manejo de errores
    try:
        tokenized = tokenizer(
            prompt, 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH, 
            padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    except Exception as e:
        print(f"Error tokenizando: {e}")
        # Proporcionar un tensor vacío como fallback
        return {
            "input_ids": torch.zeros(MAX_SEQ_LENGTH, dtype=torch.long),
            "attention_mask": torch.zeros(MAX_SEQ_LENGTH, dtype=torch.long),
            "labels": torch.zeros(MAX_SEQ_LENGTH, dtype=torch.long)
        }

# Función principal
def main():
    # Limpiar memoria antes de comenzar
    clear_memory()
    
    # Ruta al archivo JSON
    dataset_path = "breast_cancer_dataset.json"
    
    # Cargar el dataset
    print(f"Cargando dataset desde {dataset_path}")
    dataset = load_dataset(dataset_path)
    print(f"Dataset cargado con {len(dataset)} ejemplos")
    
    # Cargar el tokenizador
    print(f"Cargando tokenizador desde {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    
    # Configuración del modelo - IMPORTANTE: Usamos CPU para el entrenamiento
    # Esto resuelve el problema de gradientes en MPS
    model_device = "cpu"  # Forzar CPU para compatibilidad con gradientes
    
    print(f"Cargando modelo base en {model_device}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Importante: usar float32 para CPU
    )
    
    # IMPORTANTE: Asegurarnos de que el modelo tiene requires_grad=True
    for param in model.parameters():
        param.requires_grad = True
    
    # Liberar memoria después de cargar el modelo
    clear_memory()
    
    # Configuración de LoRA - reducir parámetros para ahorrar memoria
    # Módulos específicos para phi-2
    target_modules = ["q_proj", "k_proj", "v_proj", "dense"]
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        inference_mode=False,
    )
    
    # Obtener el modelo con LoRA
    print("Aplicando configuración LoRA al modelo...")
    model = get_peft_model(model, lora_config)
    
    # Verificar parámetros entrenables
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
    # IMPORTANTE: Volver a verificar que requires_grad esté activo
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
    
    # Liberar memoria antes de tokenizar
    clear_memory()
    
    # Tokenizar el dataset
    print("Tokenizando el dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_dataset(x, tokenizer),
        remove_columns=dataset.column_names,
        batched=False
    )
    
    # Configuración del entrenamiento optimizada para CPU
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=5,
        # IMPORTANTE: No usar fp16 ni gradient_checkpointing en CPU
        fp16=False,
        optim="adamw_torch",
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        # Desactivar gradient checkpointing en CPU
        gradient_checkpointing=False,
        # Otras optimizaciones
        remove_unused_columns=False,
        logging_first_step=True,
        no_cuda=True,  # Forzar uso de CPU
    )
    
    # Iniciar el entrenamiento
    print("Iniciando entrenamiento...")
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
    )
    
    # IMPORTANTE: Verificar que los optimizadores estén correctamente configurados
    print("Preparando optimizadores...")
    
    # Entrenar el modelo
    trainer.train()
    
    # Guardar el modelo
    print(f"Guardando modelo en {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Modelo guardado exitosamente en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()