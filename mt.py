import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, AutoAdapterModel
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import get_linear_schedule_with_warmup
import torch

MODEL = 't5-base'

def load_adapter(model_path, adapter_path, adapter_name):
    # Load the original model architecture
    model = AutoAdapterModel.from_pretrained(model_path) if len(adapter_path)>0 else T5ForConditionalGeneration.from_pretrained(model_path)
    if (len(adapter_path)>0):
        model.load_adapter(adapter_path)
    else:
        model.add_adapter(adapter_name)
    model.train_adapter(adapter_name)
    # Set the active adapter for inference
    model.set_active_adapters(adapter_name)
    # model.freeze_model()
    # Load the tokenizer
    return model

tokenizerT5 = T5Tokenizer.from_pretrained(MODEL)
def decodetext(seq):
    return tokenizerT5.decode([x for x in seq if x != 1 and x != 0])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions[0], axis=2)

    # Calculating metrics
    accuracy = accuracy_score([decodetext(x) for x in labels], [decodetext(x) for x in predictions])
    precision, recall, f1, _ = precision_recall_fscore_support([decodetext(x) for x in labels], [decodetext(x) for x in predictions], average='weighted')

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }
ADAPTER_PATH = './fine_tuned_adapter_m5'
ADAPTER = 'emotion'
model = load_adapter(MODEL, '', ADAPTER)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Load the dataset
dataset = load_dataset('emotion')


# Create a function to encode the data
def label2text(label):
    # 'sadness' 'joy' 'love'  'anger' 'fear' 'surprise
    emotion_dict = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4:'fear', 5:'surprise'}  # Fill with your actual mapping
    return emotion_dict[label]


def encode(batch, tokenizer=T5Tokenizer.from_pretrained(MODEL)):
    # Create target_text which is just the emotion text for each example in the batch
    target_text = [label2text(label) for label in batch['label']]  # Convert each emotion label to text
    inputs = tokenizer(batch['text'], truncation=True, padding='max_length', return_tensors='pt')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target_text, truncation=True, padding='max_length', return_tensors='pt')
    inputs['labels'] = labels['input_ids']
    return inputs

# Encode the dataset
dataset = dataset.map(encode, batched=True)

num_train_epochs = 1
learning_rate = 1e-5
train_dataset = dataset['train']
warmup_steps = int(len(train_dataset) * num_train_epochs * 0.1)  # 10% of train data for warm-up
total_steps = len(train_dataset) * num_train_epochs

eval_dataset=dataset['validation']

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=num_train_epochs,
    evaluation_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=16,
    learning_rate=learning_rate,
    warmup_steps=warmup_steps,
    weight_decay=0.01,
    logging_dir='./logs',
    # use_mps_device=False,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
# Create a learning rate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# Define the Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, lr_scheduler),
)

trainer.train()
model.save_adapter(ADAPTER_PATH, ADAPTER)
# clear memory
del model
del trainer
torch.cuda.empty_cache()
# test_dataset=dataset['test']
# eval_results = trainer.evaluate(test_dataset)
# print(eval_results)
