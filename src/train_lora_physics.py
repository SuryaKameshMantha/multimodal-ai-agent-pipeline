# train_lora_physics.py
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

class PhysicsQADataset(Dataset):
    def __init__(self, questions, answers, tokenizer, max_length=512):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = self.questions[idx]
        a = self.answers[idx]
        q_enc = self.tokenizer(q, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        a_enc = self.tokenizer(a, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        return {'input_ids': q_enc['input_ids'].squeeze(), 'attention_mask': q_enc['attention_mask'].squeeze(), 'labels': a_enc['input_ids'].squeeze()}

def train_lora_small():
    print("🚀 Starting LoRA fine-tuning with FLAN-T5-base and small batch size")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    df = pd.read_csv('real_physics_qa.csv')
    questions = df['question'].tolist()
    answers = df['answer'].tolist()
    print(f"Loaded {len(questions)} Q/A pairs")

    model_name = 'google/flan-t5-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=['q', 'v'],
        lora_dropout=0.1,
        bias='none',
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    dataset = PhysicsQADataset(questions, answers, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    num_epochs = 3
    num_training_steps = len(dataloader) * num_epochs
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Average loss = {avg_loss:.4f}")

    model.save_pretrained('./lora_finetuned_physics')
    print("✅ LoRA model saved in ./lora_finetuned_physics")

if __name__ == '__main__':
    train_lora_small()
