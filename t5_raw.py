import pandas as pd
from simpletransformers.t5 import T5Model
from datasets import load_dataset
from sklearn.metrics import f1_score
from statistics import mean

dataset = load_dataset('emotion')
import pandas as pd
df_train = pd.DataFrame([[x['text'], str(x['label'])] for x in dataset['train']])
df_train['prefix'] = 'multilabel classification'
df_train.columns = ['input_text', 'target_text', 'prefix']
df_val = pd.DataFrame([[x['text'], str(x['label'])] for x in dataset['validation']])
df_val['prefix'] = 'multilabel classification'
df_val.columns = ['input_text', 'target_text', 'prefix']

# Still seq2seq, but a lot (encoding/decoding) is abstracted away
model_args = {
    "max_seq_length": 196,
    "train_batch_size": 16,
    "eval_batch_size": 64,
    "num_train_epochs": 1,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 15000,
    "evaluate_during_training_verbose": True,
    "use_multiprocessing": False,
    "fp16": False,
    "save_steps": -1,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    'use_mps_device': True,
    'use_cuda': False,
    'load_best_model_at_end':True,

    #"use_cuda":True
    # "wandb_project": "T5 mixed tasks - Binary, Multi-Label, Regression",
}

model = T5Model("t5", "t5-base", args=model_args, use_cuda=False)
for i in range(5):
    model.train_model(df_train, eval_data=df_val)
    ## prediction
    df_test = pd.DataFrame([[x['text'], str(x['label'])] for x in dataset['test']])
    df_test['prefix'] = 'multilabel classification'
    df_test.columns = ['input_text', 'target_text', 'prefix']
    # df_test['prediction'] = df_test.apply(lambda x: model.predict(x['prefix'] + ": " + x['input_text']), axis = 1)
    to_predict = [
        prefix + ": " + str(input_text)
        for prefix, input_text in zip(df_test["prefix"].tolist(), df_test["input_text"].tolist())
    ]

    pred = model.predict(to_predict)

    df_test['pred'] = pred
    f1score = f1_score(df_test.target_text, df_test.pred, average=None)
    print('f1 ', f1score) ## f1 for each label

    accuracy = sum(df_test.apply(lambda x: x.pred == x.target_text, axis =1))/len(df_test) ## accuracy ~90% ish
    print('acuracy ', accuracy)
    modelname = 't5_emo-' + str(i)
    model.save_model(modelname)

