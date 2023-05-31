#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "request pretrained BERT model"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2023 by Eva Seidlmayer"
__license__ = "ISC license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1 "



import numpy as np
import argparse
import torch
from BERT_training_sliding_window import AQUASSlidingBERT
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import BertTokenizer


BERT_MODEL_IDENTIFIER = "bert-base-uncased"
#BERT_MODEL_IDENTIFIER = "dmis-lab/biobert-v1.1"
max_length = 10000
#max_length = 15000

argparser = argparse.ArgumentParser()
argparser.add_argument("model")
args = argparser.parse_args()


# Load the trained model
model = AQUASSlidingBERT.from_pretrained(args.model)

# Preprocess the specific text
text = "Your specific text here"

#preprocess text
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_IDENTIFIER)
tokens = tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors= 'pt')
#tokens = torch.tensor(tokens['input_ids'])
#input_tensor = tokens.unsqueeze(0)

input_ids = tokens['input_ids']
attn_mask = tokens['attention_mask']
print("text is preprocessed")


output = model(input_ids=input_ids, attention_mask= attn_mask)

print(output)


