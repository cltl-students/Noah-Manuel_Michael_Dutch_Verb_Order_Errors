import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW


class TokenReorganizer(torch.nn.Module):
    def __init__(self, bert_model_name):
        super(TokenReorganizer, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.output_layer = torch.nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = bert_outputs[0]
        output = self.output_layer(sequence_output)
        return output


bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)


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


# Assuming you have preprocessed and tokenized your dataset
# input_ids, target_ids, and attention_masks should be torch tensors
input_ids = ...
target_ids = ...
attention_masks = ...

# Hyperparameters
batch_size = 16
num_epochs = 3
learning_rate = 5e-5
weight_decay = 1e-2

# Create a DataLoader for your dataset
dataset = TensorDataset(input_ids, target_ids, attention_masks)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, optimizer, and device
model = TokenReorganizer("bert-base-uncased")
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fine-tuning loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0

    for batch_input_ids, batch_target_ids, batch_attention_masks in dataloader:
        batch_input_ids = batch_input_ids.to(device)
        batch_target_ids = batch_target_ids.to(device)
        batch_attention_masks = batch_attention_masks.to(device)

        optimizer.zero_grad()

        logits = model(batch_input_ids, batch_attention_masks)
        loss = masked_cross_entropy_loss(logits, batch_target_ids, batch_attention_masks)
        accuracy = masked_accuracy(logits, batch_target_ids, batch_attention_masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy.item()

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")


# Assuming you have preprocessed and tokenized your validation dataset
# val_input_ids, val_target_ids, and val_attention_masks should be torch tensors
val_input_ids = ...
val_target_ids = ...
val_attention_masks = ...

# Create a DataLoader for your validation dataset
val_dataset = TensorDataset(val_input_ids, val_target_ids, val_attention_masks)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


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
