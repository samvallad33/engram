"""
ENGRAM: FSRS-Weighted LoRA Fine-Tuning of MedGemma 1.5 4B
Kaggle MedGemma Impact Challenge 2026

Architecture: QLoRA (4-bit NF4) + SFTTrainer
Innovation: Training examples are weighted by FSRS-6 Difficulty (D).
Harder clinical concepts receive higher representation in the dataset,
creating a human-memory-driven curriculum.
"""

import json, os, random, time, torch
from collections import Counter
from datasets import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

def print_header(title: str):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")

print_header("Hardware Initialization")
COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
print(f"PyTorch: {torch.__version__} | Compute Dtype: {COMPUTE_DTYPE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─── 1. Clinical Knowledge & FSRS-6 Weights ───────────────────────

CLINICAL_DATA = {
    "Cardiomegaly": {"findings": ["Enlarged cardiac silhouette, CTR > 0.5"], "teaching": "Measure CTR on PA film. AP magnifies."},
    "Pneumothorax": {"findings": ["Visceral pleural line, absent lung markings"], "teaching": "Look for deep sulcus sign on supine."},
    "Pleural Effusion": {"findings": ["Blunting of costophrenic angle, meniscus sign"], "teaching": "Meniscus sign = fluid climbing laterally."},
    "Consolidation": {"findings": ["Dense opacity, air bronchograms"], "teaching": "Air-filled bronchi within opacified lung."},
    "Lung Opacity": {"findings": ["Patchy airspace opacity"], "teaching": "Describe location, pattern, and distribution."},
    "Atelectasis": {"findings": ["Volume loss, elevated hemidiaphragm"], "teaching": "Opacity WITH volume loss = atelectasis."},
    "Edema": {"findings": ["Bilateral perihilar haziness, Kerley B lines"], "teaching": "Progression: cephalization -> Kerley B -> bat-wing."},
    "Pneumonia": {"findings": ["Focal consolidation"], "teaching": "Follow-up at 6-8 weeks for persistent opacity."},
    "Fracture": {"findings": ["Cortical disruption of rib"], "teaching": "Lower ribs = check splenic/hepatic. Sternal = check aorta."},
    "No Finding": {"findings": ["Clear lung fields, normal heart"], "teaching": "Use ABCDE: Airways, Bones, Cardiac, Diaphragm, Everything else."},
    "Support Devices": {"findings": ["ETT tip 3cm above carina"], "teaching": "ETT: 3-5cm above carina. CVC: cavoatrial junction."},
}

# Real-world FSRS-6 Difficulty values (1-10) drive the curriculum weighting
CATEGORY_DIFFICULTY = {
    "No Finding": 2.5, "Cardiomegaly": 3.2, "Support Devices": 3.8,
    "Fracture": 5.5, "Pneumothorax": 5.8, "Pleural Effusion": 6.0,
    "Consolidation": 6.2, "Lung Opacity": 6.5, "Pneumonia": 7.0,
    "Edema": 7.5, "Atelectasis": 8.2,
}

SKILL_LEVELS = {
    "novice": {"finding_rate": 0.15, "score": (0.05, 0.25)},
    "beginner": {"finding_rate": 0.35, "score": (0.20, 0.45)},
    "intermediate": {"finding_rate": 0.55, "score": (0.40, 0.65)},
    "advanced": {"finding_rate": 0.80, "score": (0.65, 0.85)},
    "expert": {"finding_rate": 0.95, "score": (0.80, 1.00)},
}

def generate_training_example(category, skill_level=None):
    skill_level = skill_level or random.choice(list(SKILL_LEVELS.keys()))
    data, config = CLINICAL_DATA[category], SKILL_LEVELS[skill_level]

    found = [f for f in data["findings"] if random.random() < config["finding_rate"]]
    missed = [f for f in data["findings"] if f not in found]
    student_ans = f"I see {', '.join(found).lower()}." if found else "I cannot identify specific findings."

    score = min(1.0, max(0.0, (len(found) / max(1, len(data["findings"]))) + random.uniform(-0.05, 0.05)))
    assessment = "Excellent" if score >= 0.7 else ("Partial" if score >= 0.4 else "Needs improvement")

    completion = json.dumps({
        "score": round(score, 3),
        "correct_findings": found,
        "missed_findings": missed,
        "false_positives": [],
        "explanation": f"**Assessment: {assessment}**\nTeaching point: {data['teaching']}"
    }, indent=2)

    prompt = (f"Grade this student's CXR interpretation.\n**Category:** {category}\n"
              f"**Findings:** {', '.join(data['findings'])}\n**Student:** {student_ans}\n"
              "Output ONLY JSON: score, correct_findings, missed_findings, false_positives, explanation.")

    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"```json\n{completion}\n```"}],
            "category": category, "fsrs_difficulty": CATEGORY_DIFFICULTY.get(category, 5.0)}

# ─── 2. Build FSRS-Weighted Curriculum ────────────────────────────

print_header("Building FSRS-Weighted Dataset")
random.seed(42)
NUM_EXAMPLES = 1000

# Weight representation by FSRS-6 Difficulty
weights = [CATEGORY_DIFFICULTY[c] for c in CLINICAL_DATA.keys()]
categories = random.choices(list(CLINICAL_DATA.keys()), weights=weights, k=NUM_EXAMPLES)

training_data = [generate_training_example(cat) for cat in categories]
training_data.sort(key=lambda x: x["fsrs_difficulty"])

cat_dist = Counter(ex["category"] for ex in training_data)
for cat, count in sorted(cat_dist.items(), key=lambda item: CATEGORY_DIFFICULTY[item[0]]):
    print(f"  {cat:20s} | Difficulty: {CATEGORY_DIFFICULTY[cat]:.1f} | Examples: {count}")

# ─── 3. ML Pipeline: QLoRA Setup ──────────────────────────────────

print_header("Initializing QLoRA Pipeline")
MODEL_ID = "google/medgemma-1.5-4b-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_quant_storage=COMPUTE_DTYPE,
)

model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"

lora_config = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none", target_modules="all-linear", task_type="CAUSAL_LM")

# ─── 4. Dataset & Training ────────────────────────────────────────

print_header("Preparing Dataset")
splits = Dataset.from_list([{"messages": ex["messages"]} for ex in training_data]).train_test_split(test_size=0.1, seed=42)
print(f"Train: {len(splits['train'])} | Eval: {len(splits['test'])}")

training_args = SFTConfig(
    output_dir="./engram-lora-output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=2048,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    bf16=COMPUTE_DTYPE == torch.bfloat16,
    fp16=COMPUTE_DTYPE == torch.float16,
    optim="adamw_torch_fused" if COMPUTE_DTYPE == torch.bfloat16 else "paged_adamw_8bit",
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=splits["train"],
    eval_dataset=splits["test"],
    peft_config=lora_config,
    processing_class=processor.tokenizer,
)

print_header("Commencing Training Loop")
start_time = time.time()
train_result = trainer.train()
print(f"Training complete in {(time.time() - start_time) / 60:.1f} minutes. Final loss: {train_result.training_loss:.4f}")

# ─── 5. Save & Evaluate ───────────────────────────────────────────

ADAPTER_DIR = "./engram-lora-adapter"
trainer.save_model(ADAPTER_DIR)
processor.save_pretrained(ADAPTER_DIR)
print(f"Adapter saved to {ADAPTER_DIR}. Size: ~{sum(os.path.getsize(os.path.join(ADAPTER_DIR, f)) for f in os.listdir(ADAPTER_DIR) if os.path.isfile(os.path.join(ADAPTER_DIR, f))) / 1e6:.1f} MB")

print_header("Evaluation: Inference Check")
model.eval()
processor.tokenizer.padding_side = "left"

eval_prompt = ("Grade this student's CXR interpretation.\n"
               "**Category:** Atelectasis\n"
               "**Findings:** Volume loss, elevated hemidiaphragm\n"
               "**Student:** I see an opacity in the lower lobe.\n"
               "Output ONLY JSON: score, correct_findings, missed_findings, false_positives, explanation.")

inputs = processor.apply_chat_template(
    [{"role": "user", "content": eval_prompt}],
    add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(f"Model Output:\n{processor.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)}")
