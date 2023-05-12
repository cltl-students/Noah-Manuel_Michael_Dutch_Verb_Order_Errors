# 10.03.2023
# Execute the fine-tuning of BERT-based model and make predictions

from train import fine_tune_bert
from predict import make_predictions_with_finetuned_model
import sys

def main(argv=None):
    """
    Fine-tune a BERT model for Semantic Role Labeling and making predictions.

    argv[0]: the number of epochs
    argv[1]: the batch size
    argv[2]: the mode (can be 'token_type_IDs' or 'flag_with_pred_token')
    argv[3]: whether or not the test set has gold labels (True or False)
    """
    if argv is None:
        argv = sys.argv
    epochs = int(argv[0])
    batch_size = int(argv[1])
    mode = argv[2]
    has_gold = bool(argv[3])
    
    fine_tune_bert(epochs=epochs, batch_size=batch_size, mode=mode)
    make_predictions_with_finetuned_model(batch_size=batch_size, load_epoch=epochs, has_gold=has_gold, mode=mode)
    
if __name__ == '__main__':
    main(sys.argv[1:])
