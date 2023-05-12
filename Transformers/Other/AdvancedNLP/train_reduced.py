# 07.03.2023
# Fine-tune a BERT-based model for SRL
# This code was taken from the code provided for the class ML for NLP at VU Amsterdam and has been adapted
# It only contains the absolutely necessary steps to fine-tune the BERT model

import random
import bert_utils_reduced as utils
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertForTokenClassification, AdamW, BertTokenizer, get_linear_schedule_with_warmup


def fine_tune_bert(epochs: int = 5, batch_size: int = 4, mode: str = 'token_type_IDs'):
    """
    Fine-tune a BERT model for Semantic Role Labeling.

    :param int epochs: the number of epochs (how many times the model should be trained)
    :param int batch_size: the batch size (the number of instances that are processed before the model is updated)
    :param str mode: the method of fine_tuning ('token_type_IDs' or 'flag_with_pred_token')
    :return: None, but saves model to file 
    """
    assert mode == 'token_type_IDs' or mode == 'flag_with_pred_token', 'Mode for training the model wrongly specified.'

    # Initialize Hyperparameters
    EPOCHS = epochs  # How many times the model should be trained with the data of the whole training set
    BERT_MODEL_NAME = "bert-base-multilingual-cased"  # The exact BERT model you want to use
    SEED_VAL = 1234500  # Set a fixed random seed, ensures that model performance is not affected by random
    # initialization of parameters and sampling
    SEQ_MAX_LEN = 256  # The maximum length of a sequence that can be used as input, shorter and longer sequences are
    # padded/clipped to fit this length
    GRADIENT_CLIP = 1.0  # Largest gradient value, larger values get clipped to 1.0
    LEARNING_RATE = 1e-5  # The learning rate
    BATCH_SIZE = batch_size  # The batch size, after 4 instances the parameters will be updated
    PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index  # == -100, label ID for padded tokens

    # Define filepaths
    TRAIN_DATA_PATH = "Data/mini_test.json"
    LABELS_FILENAME = "saved_models/MY_BERT_SRL/label2index.json"

    # Initialize Random seeds and validate if there's a GPU available...
    device = torch.device("cpu")
    random.seed(SEED_VAL)
    np.random.seed(SEED_VAL)
    torch.manual_seed(SEED_VAL)
    torch.cuda.manual_seed_all(SEED_VAL)

    # Load Training Dataset
    # Initialize Tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, do_basic_tokenize=False)

    # Load Train Dataset
    train_data, train_labels, label2index = utils.read_json_srl(TRAIN_DATA_PATH, mode)
    utils.save_label_dict(label2index, filename=LABELS_FILENAME)
    index2label = {v: k for k, v in label2index.items()}

    train_inputs, train_masks, train_labels, token_type_IDs = \
        utils.data_to_tensors(train_data, tokenizer, max_len=SEQ_MAX_LEN, labels=train_labels, label2index=label2index,
                              pad_token_label_id=PAD_TOKEN_LABEL_ID)

    # Create the DataLoader for our training set.
    if mode == 'token_type_IDs':
        train_data = TensorDataset(train_inputs, train_masks, train_labels, token_type_IDs)
    else:
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    # Initialize Model Components
    model = BertForTokenClassification.from_pretrained(BERT_MODEL_NAME, num_labels=len(label2index))
    model.config.finetuning_task = 'token-classification'
    model.config.id2label = index2label
    model.config.label2id = label2index

    # Create optimizer and the learning rate scheduler.
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training Cycle (Fine-tuning)
    for epoch_i in range(1, EPOCHS + 1):

        # Perform one full pass over the training set.
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            if mode == 'token_type_IDs':
                b_token_type_IDs = batch[3].to(device)

            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch)
            if mode == 'token_type_IDs':
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels,
                                token_type_ids=b_token_type_IDs)
            else:
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)  # Computes the output
            # on the current batch, including the loss
            loss = outputs[0]  # Extracts the loss value from the output

            # Perform a backward pass to calculate the gradients.
            loss.backward()  # Performs backward pass to compute the gradients of the loss function

            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

            # Update parameters
            optimizer.step()
            scheduler.step()

        # Save Checkpoint for this Epoch
        utils.save_model(f"saved_models/MY_BERT_SRL_reduced_flag/EPOCH_{epoch_i}", {"args": []}, model, tokenizer)


if __name__ == '__main__':
    fine_tune_bert()
