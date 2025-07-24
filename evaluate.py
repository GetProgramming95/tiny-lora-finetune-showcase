"""
Evaluiert ein feinjustiertes LoRA-Modell auf einem kleinen manuellen Testset.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Modell & Tokenizer laden (Name wie beim Finetuning verwenden)
MODEL_NAME = "tiny-lora-model"  # ggf. Pfad anpassen (z. B. "./output")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

# CPU-Modus für Kompatibilität
device = torch.device("cpu")
model.to(device)

# Kleines Testset (Eingabe, erwarteter Output)
test_samples = [
    {
        "input": "Was ist künstliche Intelligenz?",
        "expected": "Künstliche Intelligenz ist ein Teilgebiet der Informatik..."
    },
    {
        "input": "Nenne ein Beispiel für maschinelles Lernen.",
        "expected": "Ein Beispiel ist die Bilderkennung durch neuronale Netze..."
    }
]

# Funktion zur Generierung
def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Evaluation ausgeben
print(" Evaluation des finetuned LoRA-Modells")
for sample in test_samples:
    print(f" Frage: {sample['input']}")
    result = generate_answer(sample["input"])
    print(f" Antwort: {result}")
    print(f" Erwartet: {sample['expected']}")
    print("-" * 50)
