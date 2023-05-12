# Noah-Manuel Michael
# Created: 12.05.2023
# Last updated: 12.05.2023
# test the fine-tuned detection transformer model

import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load your fine-tuned model
model_path = 'finetuned_bertje_sequence_classification'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')

# Use the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Test sentence
sentence = "Ik dat meisje weet bent je een."

# Tokenize and create input tensors
inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Set the model to evaluation mode
model.eval()

# Forward pass
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)

# Get the predicted label
logits = outputs.logits
_, predicted_label = torch.max(logits, dim=1)

print(f"Predicted label: {predicted_label.item()}")
