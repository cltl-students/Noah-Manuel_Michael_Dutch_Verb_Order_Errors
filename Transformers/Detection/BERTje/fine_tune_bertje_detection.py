# Noah-Manuel Michael
# Created: 11.05.2023
# Last updated: 13.06.2023
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb
# Fine-tune transformer models for word order error detection

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from utils_detection import SequenceClassificationDataset


def fine_tune_bertje_no_punc_for_detection():
    """

    :return:
    """
    df_train = pd.read_csv('train_shuffled_random_all_and_verbs_sampled_transformer.tsv', sep='\t', header=0,
                           encoding='utf-8')
    df_dev = pd.read_csv('dev_shuffled_random_all_and_verbs_sampled_transformer.tsv', sep='\t', header=0,
                         encoding='utf-8')

    # num_labels = 2
    tokenizer = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased', do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained('GroNLP/bert-base-dutch-cased')

    train_texts = [s for s in df_train['no_punc']] + \
                  [s for s in df_train['scrambled_no_punc']]
    train_labels = [1 for _ in range(int(len(train_texts) / 2))] + \
                   [0 for _ in range(int(len(train_texts) / 2))]
    val_texts = [s for s in df_dev['no_punc']] + \
                [s for s in df_dev['scrambled_no_punc']]
    val_labels = [1 for _ in range(int(len(val_texts) / 2))] + \
                 [0 for _ in range(int(len(val_texts) / 2))]

    train_dataset = SequenceClassificationDataset(train_texts, train_labels, tokenizer)
    val_dataset = SequenceClassificationDataset(val_texts, val_labels, tokenizer)

    training_args = TrainingArguments(output_dir='results_bertje_detection_no_punc',
                                      num_train_epochs=3,
                                      per_device_train_batch_size=128,
                                      per_device_eval_batch_size=128,
                                      warmup_steps=500,
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

    trainer.save_model('./finetuned_bertje_sequence_classification_no_punc')


if __name__ == '__main__':
    fine_tune_bertje_no_punc_for_detection()