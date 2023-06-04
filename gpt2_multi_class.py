
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

def load_go_emotion():
    DATA_NAME = "go_emotions" # SetFit/go_emotions smaller
    train_data = load_dataset(DATA_NAME, split="train")
    #print(train_data[0])
    eval_data = load_dataset(DATA_NAME, split="validation")
    label_list = train_data.features["labels"].feature.names
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    return train_data, eval_data, id2label, label2id

train_data, eval_data, id2label, label2id = load_go_emotion()
MODEL = 'distilgpt2'
tokenizer = AutoTokenizer.from_pretrained(MODEL,use_fast=True, problem_type="multi_label_classification")
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
print("Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(id2label), id2label=id2label, label2id=label2id,
    problem_type="multi_label_classification"
)
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
labels = train_data.features["labels"].feature.names

def preprocess_data(examples):
    # take a batch of texts
    text = examples["text"]
    # encode them
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    # add labels
    labels_batch = examples["labels"]
    # [[27]]
    # print('labels_batch ', labels_batch)
    # create numpy array of shape (batch_size, num_labels)
    labels_tensor = torch.zeros((len(text), len(labels)), dtype=torch.float)

    # fill numpy array
    for i, label_list in enumerate(labels_batch):
        for label in label_list:
            labels_tensor[i, label] = 1

    # print('labels_matrix ', labels_tensor)
    # labels_list = labels_tensor.numpy().tolist()

    encoding["labelx"] = labels_tensor
    # print('encoding after', encoding)

    return encoding
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

# {'text': "My favourite food is anything I didn't have to cook myself.", 'labels': [27], 'id': 'eebbqej'}

train_data = train_data.map(preprocess_data, batched=True, remove_columns=train_data.column_names)
train_data = train_data.rename_column("labelx", "labels")
print('train_data',train_data[0])

eval_data = eval_data.map(preprocess_data, batched=True, remove_columns=eval_data.column_names)
eval_data = eval_data.rename_column("labelx", "labels")
train_data.set_format("torch")
eval_data.set_format("torch")

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

import evaluate
accuracy = evaluate.load("accuracy")

import numpy as np

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    print('predictions', predictions)
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
                                           tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result

num_train_epochs = 1
learning_rate = 1e-5
warmup_steps = int(len(train_data) * num_train_epochs * 0.1)  # 10% of train data for warm-up
total_steps = len(train_data) * num_train_epochs
training_args = TrainingArguments(
    output_dir="go_emo_gpt",
    learning_rate=learning_rate,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=64,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    warmup_steps=warmup_steps,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=True,
)

import torch
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
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