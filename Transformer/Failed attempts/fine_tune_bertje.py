# Noah-Manuel Michael
# Created: 08.05.2023
# Last updated: 09.05.2023
# Fine-tune BERTje for token reorganization
# This script was pair programmed with Chat-GPT (v4)

import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
from torch.optim import AdamW
from utils_bertje import TokenReorganizer, masked_accuracy, masked_cross_entropy_loss, validate, get_input_ids, \
    get_pointer_labels, get_attention_masks, predict_reordered_sequence

df = pd.read_csv('../../Data/Dataset Construction/Data/scrambled_data.tsv', sep='\t', header=0, encoding='utf-8')

inputs = df['scrambled'][:10].tolist()
targets = df['original'][:10].tolist()

tokenizer = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
model = TokenReorganizer.from_pretrained('GroNLP/bert-base-dutch-cased')

# Dataset: input_ids, pointer_labels, and attention_masks should be torch tensors
input_ids = get_input_ids(inputs, tokenizer)
pointer_labels = get_pointer_labels(inputs, targets, tokenizer)
attention_masks = get_attention_masks(inputs, tokenizer)

# Fine-tuning loop with pointer mechanism
# Hyperparameters
batch_size = 16
num_epochs = 5
learning_rate = 5e-5
weight_decay = 1e-2

# Create a DataLoader for your dataset
dataset = TensorDataset(input_ids, pointer_labels, attention_masks)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, optimizer, and device
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0

    for batch_input_ids, batch_pointer_labels, batch_attention_masks in dataloader:
        batch_input_ids = batch_input_ids.to(device)
        batch_pointer_labels = batch_pointer_labels.to(device)
        batch_attention_masks = batch_attention_masks.to(device)

        optimizer.zero_grad()

        logits = model(batch_input_ids, batch_attention_masks)
        loss = masked_cross_entropy_loss(logits, batch_pointer_labels, batch_attention_masks)
        accuracy = masked_accuracy(logits, batch_pointer_labels, batch_attention_masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy.item()

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    val_input_ids = get_input_ids(inputs, tokenizer)
    val_pointer_labels = get_pointer_labels(inputs, targets, tokenizer)
    val_attention_masks = get_attention_masks(inputs, tokenizer)

    # Create a DataLoader for your validation dataset
    val_dataset = TensorDataset(val_input_ids, val_pointer_labels, val_attention_masks)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    val_accuracy = validate(model, val_dataloader, device)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    input_sequences = df['scrambled'][:10].tolist()

    predict_reordered_sequence(model, input_sequences, tokenizer)
