# Noah-Manuel Michael
# Created: 12.05.2023
# Last updated: 12.05.2023
# utils for BERTforSequenceclassification

import torch
from torch.utils.data import Dataset


class SequenceClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
        inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        if 'token_type_ids' in inputs:
            inputs['token_type_ids'] = inputs['token_type_ids'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        inputs['labels'] = torch.tensor(label, dtype=torch.long)
        return inputs
