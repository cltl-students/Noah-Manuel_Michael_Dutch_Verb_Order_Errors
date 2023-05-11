# Noah-Manuel Michael
# Created: 08.05.2023
# Last updated: 09.05.2023
# utils for fine-tuning BERTje
# This script was pair programmed with Chat-GPT (v4)

import torch
from transformers import BertModel


class TokenReorganizer(torch.nn.Module):
    def __init__(self, bert_model):
        super(TokenReorganizer, self).__init__()  # inherit attributes and methods from torch.nn.Module with super()
        self.bert = bert_model
        self.output_layer = torch.nn.Linear(self.bert.config.hidden_size, self.bert.config.max_position_embeddings)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = bert_outputs[0]
        output = self.output_layer(sequence_output)
        return output

    @classmethod
    def from_pretrained(cls, bert_model_name):
        bert_model = BertModel.from_pretrained(bert_model_name)
        return cls(bert_model)


def masked_cross_entropy_loss(logits, target, mask):
    """
    Compute the cross-entropy loss while considering the masked tokens.

    :param logits: The model's output logits. Shape: (batch_size, sequence_length, vocab_size)
    :param target: The target token indices. Shape: (batch_size, sequence_length)
    :param mask: The mask indicating valid tokens. Shape: (batch_size, sequence_length)
    :return: The masked cross-entropy loss.
    """
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    losses = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
    masked_losses = losses * mask.view(-1)
    return masked_losses.sum() / mask.sum()


def masked_accuracy(logits, target, mask):
    """
    Compute the accuracy while considering the masked tokens.

    :param logits: The model's output logits. Shape: (batch_size, sequence_length, vocab_size)
    :param target: The target token indices. Shape: (batch_size, sequence_length)
    :param mask: The mask indicating valid tokens. Shape: (batch_size, sequence_length)
    :return: The masked accuracy.
    """
    predictions = torch.argmax(logits, dim=-1)
    correct_predictions = (predictions == target) * mask
    return correct_predictions.sum().float() / mask.sum()


def validate(model, dataloader, device):
    model.eval()
    total_accuracy = 0.0

    with torch.no_grad():
        for batch_input_ids, batch_target_ids, batch_attention_masks in dataloader:
            batch_input_ids = batch_input_ids.to(device)
            batch_target_ids = batch_target_ids.to(device)
            batch_attention_masks = batch_attention_masks.to(device)

            logits = model(batch_input_ids, batch_attention_masks)
            accuracy = masked_accuracy(logits, batch_target_ids, batch_attention_masks)

            total_accuracy += accuracy.item()

    avg_accuracy = total_accuracy / len(dataloader)
    return avg_accuracy


def get_input_ids(inputs, tokenizer):
    input_ids = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")["input_ids"]
    return input_ids


def get_pointer_labels(inputs, targets, tokenizer):
    input_ids_batch = get_input_ids(inputs, tokenizer)
    target_ids_batch = get_input_ids(targets, tokenizer)
    pointer_labels_batch = []

    for inp_ids, tgt_ids in zip(input_ids_batch, target_ids_batch):
        inp_ids = inp_ids.tolist()
        tgt_ids = tgt_ids.tolist()

        pointers = [inp_ids.index(tgt_id) for tgt_id in tgt_ids if tgt_id in inp_ids]

        pointer_labels_batch.append(pointers)

    return torch.tensor(pointer_labels_batch)


def get_attention_masks(inputs, tokenizer):
    attention_masks = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")["attention_mask"]
    return attention_masks


def predict_reordered_sequence(model, input_sequences, tokenizer):
    for input_sequence in input_sequences:

        # Preprocess the input_sequence
        input_ids = get_input_ids([input_sequence], tokenizer)
        attention_masks = get_attention_masks([input_sequence], tokenizer)

        # Run the model to get predicted logits
        with torch.no_grad():
            output_logits = model(input_ids, attention_masks)

        # Convert the logits into position predictions
        predicted_positions = torch.argmax(output_logits, dim=-1)

        # Map the predicted positions to the actual input tokens
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        indexed_positions = [(i, position.item()) for i, position in enumerate(predicted_positions[0])]
        indexed_positions.sort(key=lambda x: x[1])

        reordered_tokens = [input_tokens[i] for i, _ in indexed_positions if input_tokens[i] not in ['[CLS]', '[SEP]']]

        print(' '.join(reordered_tokens).replace(' ##', ''))

    # return reordered_tokens
