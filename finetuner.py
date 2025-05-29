import os
import json
import torch
import random
import time
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from torch.optim import AdamW
from safetensors.torch import safe_open
from safetensors.torch import load_file

# ----- Parameters -----
MODEL_PATH = "models/model770M.safetensors" 
CONFIG_PATH = "json/config770M.json"
TOKENIZER_PATH = "gpt2"
INPUT_FILE = "json/cheese_dataset.jsonl"
TRAIN_FILE = "json/train.jsonl"
EVAL_FILE = "json/eval.jsonl"
SPLIT_RATIO = 0.9
BATCH_SIZE = 1
EPOCHS = 3
LR = 5e-5
MAX_LENGTH = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Split the dataset -----
print("Splitting dataset...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

random.shuffle(data)
split_index = int(len(data) * SPLIT_RATIO)
train_data = data[:split_index]
eval_data = data[split_index:]

with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    for entry in train_data:
        f.write(json.dumps(entry) + "\n")

with open(EVAL_FILE, "w", encoding="utf-8") as f:
    for entry in eval_data:
        f.write(json.dumps(entry) + "\n")

print(f"Dataset split: {len(train_data)} train, {len(eval_data)} eval")

# ----- Dataset Class -----
class CheeseDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                input_text = obj["input"]
                output_text = obj["output"]
                full = f"{input_text}\nCheese name: {output_text}"
                enc = tokenizer(full, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors="pt")
                self.samples.append({"input_ids": enc.input_ids.squeeze(0), "attention_mask": enc.attention_mask.squeeze(0)})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ----- Load Tokenizer -----
print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.pad_token = tokenizer.eos_token

# ----- Load Model from Safetensors -----
print(f"Loading model weights from {MODEL_PATH}...")

if CONFIG_PATH:
    with open(CONFIG_PATH, 'r') as f:
        config = GPT2Config.from_dict(json.load(f))
else:
    config = GPT2Config.from_pretrained("gpt2")

# Load Model on Meta
model = GPT2LMHeadModel(config).to('meta').to(torch.float16)
with safe_open(MODEL_PATH, framework="pt", device='cpu') as f:
    loaded_keys = f.keys()
    state_dict = {k: f.get_tensor(k) for k in loaded_keys}

# Load weights into the model
model = GPT2LMHeadModel(config)
model.load_state_dict(state_dict, strict=False)
model.resize_token_embeddings(len(tokenizer))
model = model.to(device)
model = torch.compile(model)
model.train()

# ----- Load Data -----
print("Preparing datasets and dataloaders...")
train_dataset = CheeseDataset(TRAIN_FILE, tokenizer)
eval_dataset = CheeseDataset(EVAL_FILE, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)

steps_per_epoch = len(train_loader)
print(f"Steps per epoch: {steps_per_epoch}")

# ----- Optimizer -----
optimizer = AdamW(model.parameters(), lr=LR)

# ----- Training Loop -----
for epoch in range(EPOCHS):
    print(f"\nStarting Epoch {epoch + 1}/{EPOCHS}...")
    model.train()
    total_loss = 0
    start_time = time.time()

    for step, batch in enumerate(train_loader):
        step_start = time.time()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        torch.cuda.empty_cache()
        optimizer.zero_grad()

        step_time = time.time() - step_start
        tokens_per_sec = input_ids.numel() / step_time
        total_loss += loss.item()

        print(f"step {step:5d} | loss: {loss.item():.6f} | lr {LR:.4e} | norm: {norm:.4f} | dt: {step_time*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

        # Evaluate every 300 steps or at last step
        if (step + 1) % 300 == 0 or (step + 1) == steps_per_epoch:
            print("Eval: ")
            model.eval()
            eval_loss = 0
            with torch.no_grad():
                for eval_batch in eval_loader:
                    input_ids_eval = eval_batch['input_ids'].to(device)
                    attn_eval = eval_batch['attention_mask'].to(device)
                    outputs_eval = model(input_ids=input_ids_eval, attention_mask=attn_eval, labels=input_ids_eval)
                    eval_loss += outputs_eval.loss.item()
            avg_eval_loss = eval_loss / len(eval_loader)
            print(f"Step {step+1} Eval Loss: {avg_eval_loss:.4f}")
            model.train()

            # Sample generation
            print("Sampling model output: ")
            prompt_text = "Cheese is a "  # example prompt
            input_ids_prompt = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
            generated = model.generate(
                input_ids_prompt,
                max_length=100,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
            output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"Sample Output:{prompt_text}{output_text}")
            print("-" * 50)

    avg_train_loss = total_loss / steps_per_epoch
    print(f"Avg Training Loss for Epoch {epoch + 1}: {avg_train_loss:.4f}")

    # Final Evaluation for epoch
    print("Running final evaluation for epoch...")
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            eval_loss += outputs.loss.item()
    avg_eval_loss = eval_loss / len(eval_loader)
    print(f"Final Eval Loss (Epoch {epoch + 1}): {avg_eval_loss:.4f}")

    # Save Checkpoint
    save_path = f"finetuned_model_epoch{epoch+1}.pt"
    torch.save({
        'model': model.state_dict(),
        'config': config.to_dict(),
        'step': epoch,
        'val_loss': avg_eval_loss
    }, save_path)
    print(f"Saved checkpoint to {save_path}")

