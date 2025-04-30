import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# Configuración
BASE_MODEL_ID = "microsoft/phi-2"  # O el modelo base que hayas usado
LORA_MODEL_PATH = "./lora_phi2_output"  # Ruta a tu modelo LoRA
OUTPUT_DIR = "./merged_model"  # Ruta donde guardar el modelo combinado

# Crear directorio de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar el modelo base
print(f"Cargando modelo base: {BASE_MODEL_ID}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float32,
    trust_remote_code=True
)

# Cargar tokenizador
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    trust_remote_code=True
)

# Cargar y combinar con adaptadores LoRA
print(f"Cargando adaptadores LoRA desde: {LORA_MODEL_PATH}")
model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)

# Combinar los pesos (merge)
print("Combinando modelo base con adaptadores LoRA...")
model = model.merge_and_unload()

# Guardar el modelo combinado
print(f"Guardando modelo combinado en: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("¡Modelo combinado guardado exitosamente!")