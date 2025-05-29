import os
import json
import torch
import random
import time
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from torch.optim import AdamW
from safetensors.torch import safe_open
from torch.amp import autocast, GradScaler

# ----- Parameters -----
MODEL_PATH = "models/model770M.safetensors"
CONFIG_PATH = "json/config770M.json"
MODEL_CACHE_PT = "cached_model.pt"
TOKENIZER_PATH = "gpt2"
INPUT_FILE = "json/cheese_dataset.jsonl"
TRAIN_FILE = "json/train.jsonl"
EVAL_FILE = "json/eval.jsonl"
SPLIT_RATIO = 0.9
BATCH_SIZE = 1
EPOCHS = 3
LR = 5e-5
MAX_LENGTH = 512
GRAD_ACCUM_STEPS = 2
EVAL_INTERVAL = 300
SAMPLE_PROMPT = "Cheese is a "
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Dataset Split -----
if not os.path.exists(TRAIN_FILE) or not os.path.exists(EVAL_FILE):
    print("Splitting dataset...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    random.shuffle(data)
    split_idx = int(len(data) * SPLIT_RATIO)
    train_data, eval_data = data[:split_idx], data[split_idx:]
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        for entry in eval_data:
            f.write(json.dumps(entry) + "\n")
    print(f"Dataset split: {len(train_data)} train, {len(eval_data)} eval")
else:
    print("🟡 Using cached dataset split...")

# ----- Dataset Class -----
class CheeseDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                full = f"{obj['input']}\nCheese name: {obj['output']}"
                enc = tokenizer(full, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors="pt")
                self.samples.append({"input_ids": enc.input_ids.squeeze(0), "attention_mask": enc.attention_mask.squeeze(0)})

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# ----- Load Tokenizer -----
tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.pad_token = tokenizer.eos_token

# ----- Load Config -----
with open(CONFIG_PATH, 'r') as f:
    config = GPT2Config.from_dict(json.load(f))

# ----- Load Model -----
if os.path.exists(MODEL_CACHE_PT):
    print("🟢 Loading cached PyTorch model...")
    model = GPT2LMHeadModel(config)
    model.load_state_dict(torch.load(MODEL_CACHE_PT))
    print("✅ Model Loaded")
else:
    print("🔵 Loading model from safetensors...")
    model = GPT2LMHeadModel(config).to('meta').to(torch.float16)
    with safe_open(MODEL_PATH, framework="pt", device='cpu') as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    model = GPT2LMHeadModel(config)
    model.load_state_dict(state_dict, strict=False)
    print("✅ Model Loaded")
    torch.save(model.state_dict(), MODEL_CACHE_PT)
    print(f"✅ Model cached to {MODEL_CACHE_PT}")

model.resize_token_embeddings(len(tokenizer))
model = model.to(device)
model.train()

# ----- Load Data -----
train_dataset = CheeseDataset(TRAIN_FILE, tokenizer)
eval_dataset = CheeseDataset(EVAL_FILE, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)
steps_per_epoch = len(train_loader)
print(f"Steps per epoch: {steps_per_epoch}")

# ----- Optimizer and AMP -----
optimizer = AdamW(model.parameters(), lr=LR)
scaler = GradScaler(enabled=torch.cuda.is_available())
torch.set_float32_matmul_precision("high")

# ----- Training Loop -----
for epoch in range(EPOCHS):
    print(f"\n🚀 Epoch {epoch + 1}/{EPOCHS}")
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        step_start = time.time()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == steps_per_epoch:
            scaler.unscale_(optimizer)
            with torch.no_grad():
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        step_time = time.time() - step_start
        tokens_per_sec = input_ids.numel() / step_time
        total_loss += loss.item() * GRAD_ACCUM_STEPS
        print(f"Step {step+1} | Loss: {loss.item() * GRAD_ACCUM_STEPS:.4f} | lr {LR:.4e} | dt: {step_time*1000:.2f} | tok/sec: {tokens_per_sec:.2f}")

        # Eval every X steps
        if (step + 1) % EVAL_INTERVAL == 0 or (step + 1) == steps_per_epoch or step==0:
            print("Beginning Eval")
            model.eval()
            eval_loss = 0
            with torch.no_grad():
                for eval_batch in eval_loader:
                    ids = eval_batch['input_ids'].to(device)
                    mask = eval_batch['attention_mask'].to(device)
                    with autocast(device_type='cuda'):
                        out = model(input_ids=ids, attention_mask=mask, labels=ids)
                    eval_loss += out.loss.item()
            avg_eval = eval_loss / len(eval_loader)
            print(f"\nEval Loss: {avg_eval:.4f}")

            # Sample from model
            print("Sample Output:")
            sample_ids = tokenizer.encode(SAMPLE_PROMPT, return_tensors="pt").to(device)
            with torch.no_grad():
                sample = model.generate(sample_ids, max_length=100, pad_token_id=tokenizer.eos_token_id, attention_mask=attention_mask)
            print(tokenizer.decode(sample[0], skip_special_tokens=True))
            print("-" * 50)
            model.train()

    print(f"Epoch {epoch+1} is Complete | Avg Train Loss: {total_loss / steps_per_epoch:.4f}")

    # Save checkpoint
    save_path = f"finetuned_epoch{epoch+1}.pt"
    torch.save({
        'model': model.state_dict(),
        'config': config.to_dict(),
        'step': epoch,
    }, save_path)
    print(f"Saved checkpoint to {save_path}")