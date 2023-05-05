# 20.01.2023
# Lisa's code experiment with transformer models

from transformers import BertTokenizer, BertLMHeadModel, BertConfig, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
import torch
import math

model_name = 'GroNLP/gpt2-small-dutch'

tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
config = GPT2Config.from_pretrained(model_name)
config.is_decoder = True
model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
model.eval()

sentences = ["Ik wou dat ik een kind was.", "Ik wou dat ik was een kind.",
             "Ik wou dat ik mijn moeder nog een keer kon zien.", "Ik wou dat ik mijn moeder zien nog een keer kon."]


def calculate_perplexity(sentence, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to('cpu')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)


for sent in sentences:
    ppl = calculate_perplexity(sent, model, tokenizer)

    print(sent)
    print(ppl)
    print()
