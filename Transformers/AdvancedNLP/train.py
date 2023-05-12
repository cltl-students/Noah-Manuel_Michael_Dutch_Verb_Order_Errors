# 02.03.2023
# Fine-tune a BERT-based model for SRL
# This code was taken from the code provided for the class ML for NLP at VU Amsterdam and has been adapted

import logging
import os
import random
import sys
import time
import bert_utils as utils
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertForTokenClassification, AdamW
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup


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
    GPU_RUN_IX = 0
    SEED_VAL = 1234500  # Set a fixed random seed, ensures that model performance is not affected by random
    # initialization of parameters and sampling
    SEQ_MAX_LEN = 256  # The maximum length of a sequence that can be used as input, shorter and longer sequences are
    # padded/clipped to fit this length
    PRINT_INFO_EVERY = 10  # Print status only every X batches
    GRADIENT_CLIP = 1.0  # Largest gradient value, larger values get clipped to 1.0
    LEARNING_RATE = 1e-5  # The learning rate
    BATCH_SIZE = batch_size  # The batch size, after 4 instances the parameters will be updated
    PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index  # == -100, label ID for padded tokens

    # Define filepaths
    TRAIN_DATA_PATH = "Data/mini_train.json"  # or path to full train set
    DEV_DATA_PATH = "Data/mini_dev.json"  # or path to full train/dev set
    SAVE_MODEL_DIR = "saved_models/MY_BERT_SRL/"
    LABELS_FILENAME = f"{SAVE_MODEL_DIR}/label2index.json"
    LOSS_TRN_FILENAME = f"{SAVE_MODEL_DIR}/Losses_Train_{EPOCHS}.json"
    LOSS_DEV_FILENAME = f"{SAVE_MODEL_DIR}/Losses_Dev_{EPOCHS}.json"

    if not os.path.exists(SAVE_MODEL_DIR):
        os.makedirs(SAVE_MODEL_DIR)

    # Initialize Random seeds and validate if there's a GPU available...
    device, USE_CUDA = utils.get_torch_device(GPU_RUN_IX)
    random.seed(SEED_VAL)
    np.random.seed(SEED_VAL)
    torch.manual_seed(SEED_VAL)
    torch.cuda.manual_seed_all(SEED_VAL)

    # Record everything inside a Log File
    console_hdlr = logging.StreamHandler(sys.stdout)
    file_hdlr = logging.FileHandler(filename=f"{SAVE_MODEL_DIR}/BERT_TokenClassifier_train_{EPOCHS}.log")
    logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])
    logging.info("Start Logging")

    # Load Training and Validation  Datasets
    # Initialize Tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, do_basic_tokenize=False)

    # Load Train Dataset
    train_data, train_labels, label2index = utils.read_json_srl(TRAIN_DATA_PATH, mode)
    utils.save_label_dict(label2index, filename=LABELS_FILENAME)
    index2label = {v: k for k, v in label2index.items()}

    train_inputs, train_masks, train_labels, seq_lengths, token_type_IDs = \
        utils.data_to_tensors(train_data, tokenizer, max_len=SEQ_MAX_LEN, labels=train_labels, label2index=label2index,
                              pad_token_label_id=PAD_TOKEN_LABEL_ID)

    # Create the DataLoader for our training set.
    if mode == 'token_type_IDs':
        train_data = TensorDataset(train_inputs, train_masks, train_labels, seq_lengths, token_type_IDs)
    else:
        train_data = TensorDataset(train_inputs, train_masks, train_labels, seq_lengths)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    # Load Dev Dataset
    dev_data, dev_labels, _ = utils.read_json_srl(DEV_DATA_PATH, mode)
    dev_inputs, dev_masks, dev_labels, dev_lens, dev_token_type_IDs = \
        utils.data_to_tensors(dev_data, tokenizer, max_len=SEQ_MAX_LEN, labels=dev_labels, label2index=label2index,
                              pad_token_label_id=PAD_TOKEN_LABEL_ID)

    # Create the DataLoader for our Development set.
    if mode == 'token_type_IDs':
        dev_data = TensorDataset(dev_inputs, dev_masks, dev_labels, dev_lens, dev_token_type_IDs)
    else:
        dev_data = TensorDataset(dev_inputs, dev_masks, dev_labels, dev_lens)
    dev_sampler = RandomSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=BATCH_SIZE)

    # Initialize Model Components
    model = BertForTokenClassification.from_pretrained(BERT_MODEL_NAME, num_labels=len(label2index))
    model.config.finetuning_task = 'token-classification'
    model.config.id2label = index2label
    model.config.label2id = label2index
    if USE_CUDA:
        model.cuda()

    # Create optimizer and the learning rate scheduler.
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training Cycle(Fine-tunning)
    loss_trn_values, loss_dev_values = [], []

    for epoch_i in range(1, EPOCHS + 1):

        # Perform one full pass over the training set.
        logging.info("")
        logging.info('======== Epoch {:} / {:} ========'.format(epoch_i, EPOCHS))
        logging.info('Training...')

        t0 = time.time()
        total_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            if mode == 'token_type_IDs':
                b_token_type_IDs = batch[4].to(device)

            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch)
            if mode == 'token_type_IDs':
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels,
                                token_type_ids=b_token_type_IDs)
            else:
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

            loss = outputs[0]
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

            # Update parameters
            optimizer.step()
            scheduler.step()

            # Progress update
            if step % PRINT_INFO_EVERY == 0 and step != 0:
                # Calculate elapsed time in minutes.
                elapsed = utils.format_time(time.time() - t0)
                # Report progress.
                logging.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Loss: {}.'.format(step,
                                                                                                len(train_dataloader),
                                                                                                elapsed, loss.item()))

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_trn_values.append(avg_train_loss)

        logging.info("")
        logging.info("  Average training loss: {0:.4f}".format(avg_train_loss))
        logging.info("  Training Epoch took: {:}".format(utils.format_time(time.time() - t0)))

        # Validation
        # After the completion of each training epoch, measure our performance on our validation set.
        t0 = time.time()
        results, preds_list = utils.evaluate_bert_model(dev_dataloader, BATCH_SIZE, model, tokenizer, index2label,
                                                        PAD_TOKEN_LABEL_ID, prefix="Validation Set")

        loss_dev_values.append(results['loss'])
        logging.info("  Validation Loss: {0:.2f}".format(results['loss']))
        logging.info("  Precision: {0:.2f} || Recall: {1:.2f} || F1: {2:.2f}".format(results['precision'] * 100,
                                                                                     results['recall'] * 100,
                                                                                     results['f1'] * 100))
        logging.info("  Validation took: {:}".format(utils.format_time(time.time() - t0)))

        # Save Checkpoint for this Epoch
        utils.save_model(f"{SAVE_MODEL_DIR}/EPOCH_{epoch_i}", {"args": []}, model, tokenizer)

    utils.save_losses(loss_trn_values, filename=LOSS_TRN_FILENAME)
    utils.save_losses(loss_dev_values, filename=LOSS_DEV_FILENAME)
    logging.info("")
    logging.info("Training complete!")


if __name__ == '__main__':
    fine_tune_bert()
