# Noah-Manuel Michael
# Created: 11.05.2023
# Last updated: 12.05.2023
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb
# Fine-tune RobBERT for incorrect word order detection (complete random scramble, only one set of scrambled or many
# different sets of scrambled tokens from the same sentence)
# Try to fine-tune on only verbs scrambled for incorrect verb order detection, one sentence can be used to train as many
# times as there are permutations for the verbs
# For now, first attempt to fine tune with one set of randomly scrambled sentences

# First fine-tune a model for classification
# Next step will be to modify the data creation

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from utils_detection import SequenceClassificationDataset

df = pd.read_csv('../../Data/Dataset Construction/Data/scrambled_data.tsv', sep='\t', header=0, encoding='utf-8')

# num_labels = 2
tokenizer = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
model = BertForSequenceClassification.from_pretrained('GroNLP/bert-base-dutch-cased')

train_texts = [t for t in df['original'][:100]] + \
              [t for t in df['scrambled'][:100]]
train_labels = [1 for _ in range(int(len(train_texts)/2))] + \
               [0 for _ in range(int(len(train_texts)/2))]
val_texts = [t for t in df['original'][100:150]] + \
            [t for t in df['scrambled'][100:150]]
val_labels = [1 for _ in range(int(len(val_texts)/2))] + \
             [0 for _ in range(int(len(val_texts)/2))]

train_dataset = SequenceClassificationDataset(train_texts, train_labels, tokenizer)
val_dataset = SequenceClassificationDataset(val_texts, val_labels, tokenizer)

training_args = TrainingArguments(output_dir='results',
                                  num_train_epochs=3,
                                  per_device_train_batch_size=16,
                                  per_device_eval_batch_size=16,
                                  warmup_steps=0,
                                  weight_decay=0.01,
                                  save_strategy='epoch',
                                  evaluation_strategy='epoch',
                                  load_best_model_at_end=True,
                                  report_to=[])

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_dataset,
                  eval_dataset=val_dataset)

trainer.train()

trainer.save_model('./finetuned_bertje_sequence_classification')
