"""
Kompaktes LoRA-Finetuning mit HuggingFace & PEFT
Feintuning eines Mini-LLM-Modells auf einfache Textdaten mit LoRA
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import torch
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_NAME = "tiiuae/falcon-rw-1b"  # Oder "EleutherAI/pythia-70m"
DATA_PATH = "data/fine_tune_data.txt"

# 1. Daten vorbereiten
def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

def prepare_data(tokenizer):
    raw_dataset = load_dataset("text", data_files={"train": DATA_PATH})
    tokenized = raw_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    return tokenized["train"]

# 2. LoRA konfigurieren
def apply_lora(model):
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, config)

# 3. Training starten
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)

    model = apply_lora(model)
    dataset = prepare_data(tokenizer)

    args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=50,
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

if __name__ == "__main__":
    main()
