# Noah-Manuel Michael
# Created: 11.05.2023
# Last updated: 11.05.2023
# Fine-tune BERTje for token reorganization
# This script was pair programmed with Chat-GPT (v4)

from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertTokenizer, Trainer, TrainingArguments
from utils_bertje_3 import assign_position_ids, ReorderDataset

# Initializing the tokenizer
tokenizer = BertTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")

# Sentences
correct_sentences = ["De snelle bruine vos springt over de luie hond."]
scrambled_sentences = ["snelle De bruine over springt luie hond de vos."]

# Create the dataset
reorder_dataset = ReorderDataset(correct_sentences, scrambled_sentences, tokenizer)

# Create training arguments and the Trainer
training_args = TrainingArguments(
    output_dir="results_bertje_detection",
    num_train_epochs=1,  # You should use more epochs for better results_bertje_detection
    per_device_train_batch_size=8,
    logging_dir="./logs",
    save_strategy="epoch"
)

model = BertForMaskedLM.from_pretrained("GroNLP/bert-base-dutch-cased")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=reorder_dataset,
)

# Train the model
trainer.train()
