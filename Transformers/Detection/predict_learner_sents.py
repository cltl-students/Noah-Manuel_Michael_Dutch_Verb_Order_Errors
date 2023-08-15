# Noah-Manuel Michael
# Created: 07.08.2023
# Last updated: 07.08.2023
# Get predictions on learner sentences

import torch
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, \
    RobertaForSequenceClassification, GPT2Tokenizer, GPT2ForSequenceClassification


def get_predictions_transformers_learner_sents(general_model_path):
    """

    :param general_model_path:
    :return:
    """
    # Load your fine-tuned model
    learner_sents = []

    with open('../../Data/Annotation/Data/leerder_sents_no_punc_for_testing.tsv') as infile:
        for line in infile.readlines():
            learner_sents.append(line.strip())

    for model_iteration in ['', '2', '3']:
        model_path = general_model_path + model_iteration

        if 'bertje' in model_path:
            if 'verbs' in model_path:
                model_name = 'bertje_Verbs'
            else:
                model_name = 'bertje_Rand'
            model = BertForSequenceClassification.from_pretrained(model_path)
            tokenizer = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased', do_lower_case=True)
        elif 'robbert' in model_path:
            if 'verbs' in model_path:
                model_name = 'robbert_Verbs'
            else:
                model_name = 'robbert_Rand'
            model = RobertaForSequenceClassification.from_pretrained(model_path)
            tokenizer = RobertaTokenizer.from_pretrained('pdelobelle/robbert-v2-dutch-base', do_lower_case=True)
        else:
            if 'verbs' in model_path:
                model_name = 'gpt2_Verbs'
            else:
                model_name = 'gpt2_Rand'
            model = GPT2ForSequenceClassification.from_pretrained(model_path)
            tokenizer = GPT2Tokenizer.from_pretrained('GroNLP/gpt2-small-dutch', do_lower_case=True)
            tokenizer.pad_token = tokenizer.eos_token

        # Use the GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Tokenize and create input tensors
        inputs = tokenizer(learner_sents, return_tensors="pt", padding=True, truncation=True, max_length=64)
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
        with open(f'Predictions_Learn/predictions_{model_name}_on_Learn{model_iteration}.txt', 'w') as outfile:
            for label in predicted_labels:
                if label == 1:
                    outfile.write('correct\n')
                else:
                    outfile.write('incorrect\n')

        print(f'Predictions of {model_name.title() + model_iteration} on Learn written to file.')


if __name__ == '__main__':
    for gen_model_path in ['BERTje/finetuned_bertje_sequence_classification_no_punc',
                           'BERTje/finetuned_bertje_sequence_classification_verbs_no_punc',
                           'RobBERT/finetuned_robbert_sequence_classification_no_punc',
                           'RobBERT/finetuned_robbert_sequence_classification_verbs_no_punc',
                           'GPT-2/finetuned_gpt2_sequence_classification_no_punc',
                           'GPT-2/finetuned_gpt2_sequence_classification_verbs_no_punc',
                           ]:
        get_predictions_transformers_learner_sents(gen_model_path)
