# 07.03.2023
# Predict SR with a loaded model
# This code was taken from the code provided for the class ML for NLP at VU Amsterdam and has been adapted
# It only includes the absolutely necessary steps to make predictions with a fine-tuned BERT model

import bert_utils_reduced as utils
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertForTokenClassification, BertTokenizer, pipeline


def make_predictions_with_finetuned_model(batch_size: int = 4, load_epoch: int = 1, has_gold: bool = True, 
                                          mode: str = 'token_type_IDs'):
    """
    Make predictions for an SRL task with the BERT model created in train_reduced.py.

    :param int batch_size: the batch size (the number of instances that are processed before the model is updated) 
    :param int load_epoch: the epoch we wish to load 
    :param bool has_gold: whether or not the test set has gold labels
    :param str mode: the method of fine_tuning ('token_type_IDs' or 'flag_with_pred_token')
    :return: None, but saves predictions to file
    """
    assert mode == 'token_type_IDs' or mode == 'flag_with_pred_token', 'Mode for training the model wrongly specified.'

    # Use Fine-tuned Model for Predictions
    FILE_HAS_GOLD = has_gold
    SEQ_MAX_LEN = 256
    BATCH_SIZE = batch_size
    PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index  # -100
    LOAD_EPOCH = load_epoch

    TEST_DATA_PATH = "Unpermuted_Datasets/mini_test.json"
    INPUTS_PATH = f"saved_models/MY_BERT_SRL/{LOAD_EPOCH}/model_inputs.txt"
    OUTPUTS_PATH = f"saved_models/MY_BERT_SRL/{LOAD_EPOCH}/model_outputs.txt"

    # Load Pre-trained Model
    model, tokenizer = utils.load_model(BertForTokenClassification, BertTokenizer,
                                        f"saved_models/MY_BERT_SRL/{LOAD_EPOCH}")
    label2index = utils.load_label_dict("saved_models/MY_BERT_SRL/label2index.json")
    index2label = {v: k for k, v in label2index.items()}

    # Load File for Predictions
    test_data, test_labels, _ = utils.read_json_srl(TEST_DATA_PATH, mode)
    prediction_inputs, prediction_masks, gold_labels, token_type_IDs = \
        utils.data_to_tensors(test_data, tokenizer, max_len=SEQ_MAX_LEN, labels=test_labels, label2index=label2index)

    # Make Predictions
    if FILE_HAS_GOLD:
        if mode == 'token_type_IDs':
            prediction_data = TensorDataset(prediction_inputs, prediction_masks, gold_labels, token_type_IDs)
        else:
            prediction_data = TensorDataset(prediction_inputs, prediction_masks, gold_labels)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE)

        results, preds_list = utils.evaluate_bert_model(prediction_dataloader, model, tokenizer,
                                                        index2label, PAD_TOKEN_LABEL_ID, full_report=True, mode=mode)

        print("Tests Loss: {0:.2f}".format(results['loss']))
        print("Precision: {0:.2f} || Recall: {1:.2f} || F1: {2:.2f}".format(results['precision'] * 100,
                                                                            results['recall'] * 100,
                                                                            results['f1'] * 100))

        with open(OUTPUTS_PATH, "w") as fout:
            with open(INPUTS_PATH, "w") as fin:
                for sent, pred in preds_list:
                    fin.write(" ".join(sent) + "\n")
                    fout.write(" ".join(pred) + "\n")

    else:
        nlp = pipeline('token-classification', model=model, tokenizer=tokenizer)
        nlp.ignore_labels = []
        with open(OUTPUTS_PATH, "w") as fout:
            with open(INPUTS_PATH, "w") as fin:
                for seq_ix, seq in enumerate(test_data):
                    sentence = " ".join(seq)
                    predicted_labels = []
                    output_obj = nlp(sentence)
                    for tok in output_obj:
                        if '##' not in tok['word']:
                            predicted_labels.append(tok['entity'])
                    fin.write(sentence + "\n")
                    fout.write(" ".join(predicted_labels) + "\n")


if __name__ == '__main__':
    make_predictions_with_finetuned_model()
