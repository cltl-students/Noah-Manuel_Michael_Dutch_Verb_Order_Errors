# Noah-Manuel Michael
# Created: 09.05.2023
# Last updated: 09.05.2023
# Fine-tune BERTje for token reorganization
# This script was pair programmed with Chat-GPT (v4)

from torch.utils.data import random_split
from transformers import BertTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from utils_bertje_2 import BertforTokenReorganization, SentenceReorderingDataset


def main():
    tokenizer = BertTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
    model = BertforTokenReorganization.from_pretrained("GroNLP/bert-base-dutch-cased")

    dataset = SentenceReorderingDataset('../../../Data/Dataset Construction/Data/readability_data_scrambled.tsv', tokenizer)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    training_args = TrainingArguments(
        output_dir="results_bertje_detection",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir="./logs",
        logging_steps=500,
        save_strategy="epoch",
        save_total_limit=3,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    main()
