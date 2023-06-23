# Noah-Manuel Michael
# Created: 23.06.2023
# Last updated: 23.06.2023
# get gpt2 predictions on VT

import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification


def get_predictions_gpt2_VT(model_path):
    """

    :param model_path:
    :return:
    """
    # Load your fine-tuned model
    model = GPT2ForSequenceClassification.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained('GroNLP/gpt2-small-dutch', do_lower_case=True)
    tokenizer.pad_token = tokenizer.eos_token

    df_test = pd.read_csv('../../../Data/Dataset_Construction/Permuted_Datasets/'
                          'test_shuffled_random_all_and_verbs_and_tendencies.tsv', sep='\t', encoding='utf-8', header=0)
    test_set = [s for s in df_test['tendencies_no_punc']]

    # Use the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Tokenize and create input tensors
    inputs = tokenizer(test_set, return_tensors="pt", padding=True, truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Set the model to evaluation mode
    model.eval()

    # Define the batch size
    batch_size = 16

    # Predict labels in batches
    predicted_labels = []
    for i in range(0, len(input_ids), batch_size):
        batch_input_ids = input_ids[i:i + batch_size]
        batch_attention_mask = attention_mask[i:i + batch_size]

        # Forward pass
        with torch.no_grad():
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask)

        # Get the predicted label
        logits = outputs.logits
        _, batch_predicted_labels = torch.max(logits, dim=1)
        predicted_labels.extend(batch_predicted_labels.tolist())

    # Write predictions to file
    with open(f'../Predictions/predictions_gpt2_'
              f'{model_path.replace("finetuned_gpt2_sequence_classification_", "")}_VT.txt', 'w') as outfile:
        for label in predicted_labels:
            if label == 1:
                outfile.write('correct\n')
            else:
                outfile.write('incorrect\n')


if __name__ == '__main__':
    get_predictions_gpt2_VT('finetuned_gpt2_sequence_classification_no_punc')
    get_predictions_gpt2_VT('finetuned_gpt2_sequence_classification_verbs_no_punc')
