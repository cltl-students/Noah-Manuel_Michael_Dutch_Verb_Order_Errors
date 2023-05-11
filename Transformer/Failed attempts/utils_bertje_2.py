# Noah-Manuel Michael
# Created: 09.05.2023
# Last updated: 09.05.2023
# utils for fine-tuning BERTje
# This script was pair programmed with Chat-GPT (v4)

import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import Datase
from transformers import BertForMaskedLM


class BertforTokenReorganization(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        # Modify the prediction_scores to restrict the model to only draw from the input words
        input_ids_set = set(input_ids.tolist()[0])
        vocab_size = prediction_scores.shape[-1]
        mask = torch.tensor([[1 if i in input_ids_set else 0 for i in range(vocab_size)]],
                            dtype=torch.float).to(prediction_scores.device)
        prediction_scores = prediction_scores * mask

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, vocab_size), labels.view(-1))
            outputs = (masked_lm_loss,) + outputs[2:]

        return outputs


class SentenceReorderingDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        df = pd.read_csv(filename, sep='\t', header=0, encoding='utf-8')
        df_reduced = df[:10].copy()
        for i, row in df_reduced.iterrows():
            self.data.append((row['original'], row['scrambled']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        correct_sentence, scrambled_sentence = self.data[idx]
        encoded_input = self.tokenizer(scrambled_sentence, return_tensors='pt', padding='max_length',
                                       truncation=True, max_length=self.max_length)
        encoded_label = self.tokenizer(correct_sentence, return_tensors='pt', padding='max_length', truncation=True,
                                       max_length=self.max_length)
        return {
            'input_ids': encoded_input['input_ids'].squeeze(0),
            'attention_mask': encoded_input['attention_mask'].squeeze(0),
            'labels': encoded_label['input_ids'].squeeze(0)
        }
