import torch
from transformers import BertTokenizer, BertForMaskedLM

MODEL_PATH = "results/checkpoint-3"  # Replace "XXXX" with the actual checkpoint number
scrambled_sentence = "Ga je huis naar?"

tokenizer = BertTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
model = BertForMaskedLM.from_pretrained(MODEL_PATH)

# Tokenize the input sentence
input_tokens = tokenizer(scrambled_sentence, return_tensors="pt")

# Get the masked language model predictions
with torch.no_grad():
    output = model(**input_tokens)

# Create a mask to only consider tokens from the input
token_ids = input_tokens["input_ids"].squeeze().tolist()
mask = torch.zeros_like(output.logits)
for i, token_id in enumerate(token_ids):
    mask[:, i, token_id] = 1

# Get the predicted token IDs by applying the mask
predicted_token_ids = torch.argmax(output.logits * mask, dim=-1)

# Convert the token IDs back to text
predicted_sentence = tokenizer.decode(predicted_token_ids[0])

print("Scrambled sentence:", scrambled_sentence)
print("Predicted sentence:", predicted_sentence)
