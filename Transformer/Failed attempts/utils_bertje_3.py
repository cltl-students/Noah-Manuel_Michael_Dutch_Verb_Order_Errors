# Noah-Manuel Michael
# Created: 11.05.2023
# Last updated: 11.05.2023
# utils for fine-tuning BERTje
# This script was pair programmed with Chat-GPT (v4)

import torch
from torch.utils.data import Dataset


class ReorderDataset(Dataset):
    def __init__(self, correct_sentences, scrambled_sentences, tokenizer):
        self.correct_sentences = correct_sentences
        self.scrambled_sentences = scrambled_sentences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.correct_sentences)

    def __getitem__(self, idx):
        correct_tokens = self.tokenizer.tokenize(self.correct_sentences[idx])
        scrambled_tokens = self.tokenizer.tokenize(self.scrambled_sentences[idx])
        scrambled_position_ids = assign_position_ids(correct_tokens, scrambled_tokens)

        # Tokenize and create input tensors
        inputs = self.tokenizer(self.scrambled_sentences[idx], return_tensors="pt")
        inputs["input_ids"] = inputs["input_ids"].squeeze(0)
        inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)
        inputs["position_ids"] = torch.tensor(scrambled_position_ids, dtype=torch.long)

        # Tokenize and create target tensors
        targets = self.tokenizer(self.correct_sentences[idx], return_tensors="pt")
        targets["input_ids"] = targets["input_ids"].squeeze(0)

        return inputs, targets["input_ids"]


def assign_position_ids(correct_tokens, scrambled_tokens):
    position_ids = []
    for token in scrambled_tokens:
        if token in correct_tokens:
            position_id = correct_tokens.index(token)
            position_ids.append(position_id)
        else:
            position_ids.append(-1)  # For tokens not found in the correct sentence, assign -1
    return position_ids


def reorder_tokens(model, tokenizer, scrambled_sentence):
    inputs = tokenizer(scrambled_sentence, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Calculate the probability of each token at each position
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Mask logits of tokens that are not part of the input sequence
    mask = torch.ones_like(logits) * float('-inf')
    for token_id in input_ids[0]:
        mask[:, :, token_id] = 0

    # Get token counts in the input sequence
    token_counts = {token_id.item(): torch.sum(input_ids == token_id).item() for token_id in input_ids[0]}

    # Reorder tokens based on the highest probability
    reordered_tokens = []
    for position in range(input_ids.shape[1]):
        masked_logits = logits.clone() + mask
        for token_id, count in token_counts.items():
            if count <= 0:
                masked_logits[:, :, token_id] = float('-inf')

        token_probs = torch.softmax(masked_logits, dim=-1)
        max_prob_index = torch.argmax(token_probs[0, position]).item()
        reordered_tokens.append(tokenizer.convert_ids_to_tokens([max_prob_index])[0])
        token_counts[max_prob_index] -= 1

    # Convert reordered tokens to a sentence
    reordered_sentence = tokenizer.convert_tokens_to_string(reordered_tokens)
    return reordered_sentence
