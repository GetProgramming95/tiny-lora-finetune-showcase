# Tiny LoRA Finetuning Showcase

Dieses Projekt demonstriert, wie man ein kleines Sprachmodell mit LoRA (Low-Rank Adaptation) effizient finetunen kann. Ziel ist es, ein besseres Verständnis für Transfer Learning, PEFT und Modellanpassung zu entwickeln – ein essenzielles Thema für AI Research Engineers und Applied NLP Professionals.

---

##  Was wird gezeigt?

- Datenaufbereitung für das Finetuning
- LoRA-Konfiguration mit `peft`
- Training eines Mini-Sprachmodells auf individuelle Daten
- Nutzung von HuggingFace `transformers`, `datasets` und `accelerate`
- Ressourcenoptimiertes Finetuning (RAM-/VRAM-freundlich)

---

##  Technologien

- `transformers` (Modelle & Tokenizer)
- `peft` (LoRA-Adapter)
- `datasets` (Datenschnittstellen)
- `accelerate` (Trainer-Optimierung)
- `torch` (Modell-Backend)

---

##  Projektstruktur

```bash
tiny-lora-finetune/
├── data/
│   └── fine_tune_data.txt    # Trainingsdaten
├── training.py               # Finetuning-Skript
├── requirements.txt          # Abhängigkeiten
├── README.md                 # Projektdoku
└── evaluation.md             # Reflexion
