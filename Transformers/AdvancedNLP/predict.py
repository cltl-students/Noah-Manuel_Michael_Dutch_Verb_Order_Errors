# 02.03.2023
# Predict SR with a loaded model
# This code was taken from the code provided for the class ML for NLP at VU Amsterdam and has been adapted

import logging
import sys
import bert_utils as utils
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForTokenClassification
from transformers import BertTokenizer
from transformers import pipeline
from torch.utils.data import SequentialSampler


def make_predictions_with_finetuned_model(batch_size: int = 4, load_epoch: int = 1, has_gold: bool = True,
                                          mode: str = 'token_type_IDs'):
    """
    Make predictions for an SRL task with the BERT model created in train.py.

    :param int batch_size: the batch size (the number of instances that are processed before the model is updated) 
    :param int load_epoch: the epoch we wish to load 
    :param bool has_gold: whether or not the test set has gold labels
    :param str mode: the method of fine_tuning ('token_type_IDs' or 'flag_with_pred_token')
    :return: None, but saves predictions to file
    """
    assert mode == 'token_type_IDs' or mode == 'flag_with_pred_token', 'Mode for training the model wrongly specified.'

    # Use Fine-tuned Model for Predictions
    GPU_IX = 0
    device, USE_CUDA = utils.get_torch_device(GPU_IX)
    FILE_HAS_GOLD = has_gold
    SEQ_MAX_LEN = 256
    BATCH_SIZE = batch_size

    TEST_DATA_PATH = "Data/mini_test.json"  # or path to full test set
    MODEL_DIR = "saved_models/MY_BERT_SRL/"
    LOAD_EPOCH = load_epoch
    INPUTS_PATH = f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/model_inputs.txt"
    OUTPUTS_PATH = f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/model_outputs.txt"
    PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index  # -100

    console_hdlr = logging.StreamHandler(sys.stdout)
    file_hdlr = logging.FileHandler(filename=f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/BERT_TokenClassifier_predictions.log")
    logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])

    # Load Pre - trained Model
    model, tokenizer = utils.load_model(BertForTokenClassification, BertTokenizer, f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}")
    label2index = utils.load_label_dict(f"{MODEL_DIR}/label2index.json")
    index2label = {v: k for k, v in label2index.items()}

    # Load File for Predictions
    test_data, test_labels, _ = utils.read_json_srl(TEST_DATA_PATH, mode)
    prediction_inputs, prediction_masks, gold_labels, seq_lens, token_type_IDs = \
        utils.data_to_tensors(test_data, tokenizer, max_len=SEQ_MAX_LEN, labels=test_labels, label2index=label2index)

    # Make Predictions
    if FILE_HAS_GOLD:
        if mode == 'token_type_IDs':
            prediction_data = TensorDataset(prediction_inputs, prediction_masks, gold_labels, seq_lens, token_type_IDs)
        else:
            prediction_data = TensorDataset(prediction_inputs, prediction_masks, gold_labels, seq_lens)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE)

        logging.info('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

        results, preds_list = utils.evaluate_bert_model(prediction_dataloader, BATCH_SIZE, model, tokenizer,
                                                        index2label, PAD_TOKEN_LABEL_ID, full_report=True,
                                                        prefix="Tests Set", mode=mode)

        logging.info("  Tests Loss: {0:.2f}".format(results['loss']))
        logging.info("  Precision: {0:.2f} || Recall: {1:.2f} || F1: {2:.2f}".format(results['precision'] * 100,
                                                                                     results['recall'] * 100,
                                                                                     results['f1'] * 100))

        with open(OUTPUTS_PATH, "w") as fout:
            with open(INPUTS_PATH, "w") as fin:
                for sent, pred in preds_list:
                    fin.write(" ".join(sent) + "\n")
                    fout.write(" ".join(pred) + "\n")

    else:
        logging.info('Predicting labels for {:,} test sentences...'.format(len(test_data)))
        if not USE_CUDA:
            GPU_IX = -1
        nlp = pipeline('token-classification', model=model, tokenizer=tokenizer, device=GPU_IX)
        nlp.ignore_labels = []
        with open(OUTPUTS_PATH, "w") as fout:
            with open(INPUTS_PATH, "w") as fin:
                for seq_ix, seq in enumerate(test_data):
                    sentence = " ".join(seq)
                    predicted_labels = []
                    output_obj = nlp(sentence)
                    # [print(o) for o in output_obj]
                    for tok in output_obj:
                        if '##' not in tok['word']:
                            predicted_labels.append(tok['entity'])
                    logging.info(f"\n----- {seq_ix + 1} -----\n{seq}\nPRED:{predicted_labels}")
                    fin.write(sentence + "\n")
                    fout.write(" ".join(predicted_labels) + "\n")


if __name__ == '__main__':
    make_predictions_with_finetuned_model()
