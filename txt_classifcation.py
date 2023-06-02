from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

MODEL ='tingtone/jq_emo_distilbert' # "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL,use_fast=True)

train_data = load_dataset("emotion", split="train")
#print(train_data[0])
eval_data = load_dataset("emotion", split="validation")

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

train_data = train_data.map(tokenize, batched=True)
#print(train_data[0])

eval_data = eval_data.map(tokenize, batched=True)
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

import evaluate
accuracy = evaluate.load("accuracy")

import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return accuracy.compute(predictions=predictions, references=labels)

# id2label = {0: "NEGATIVE", 1: "POSITIVE"}
# label2id = {"NEGATIVE": 0, "POSITIVE": 1}
id2label = {0: 'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}
label2id = {'sadness':0, 'joy':1, 'love':2, 'anger':3, 'fear':4, 'surprise':5}

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(id2label), id2label=id2label, label2id=label2id
)
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_train_epochs = 1
learning_rate = 1e-5
warmup_steps = int(len(train_data) * num_train_epochs * 0.1)  # 10% of train data for warm-up
total_steps = len(train_data) * num_train_epochs
training_args = TrainingArguments(
    output_dir="jq_emo_distilbert",
    learning_rate=learning_rate,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    warmup_steps=warmup_steps,
    load_best_model_at_end=True,
    push_to_hub=True,
    #use_mps_device=True, # mac m1
)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
# Create a learning rate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, lr_scheduler),
)
print("Training...")
trainer.train()
#trainer.push_to_hub()