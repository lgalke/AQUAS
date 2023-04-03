#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "train datatset with HealthBERT"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2023 by Eva Seidlmayer"
__license__ = "ISC license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1 "


import argparse
import pandas as pd
import torch
#from transformers import HealthBertTokenizer, HealthBertForSequenceClassification, AdamW
#from transformers import BioBertTokenizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Define the training data
def load_dataset(input_file_csv):
    df = pd.read_csv(input_file_csv, sep=',')
    df = df.sample(frac=1)
    df = df.astype(str)
    texts = df['text'].to_list()
    labels = df['category-id'].to_list()
    print('lists created')
    return texts, labels

# Set up the HealthBERT model and tokenizer
def load_model_tokenizer():
    #model = HealthBertForSequenceClassification.from_pretrained('microsoft/healthbert-base', num_labels=3)
    #tokenizer = HealthBertTokenizer.from_pretrained('microsoft/healthbert-base')

    #tokenizer = BioBertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
    model_name = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    print('model and tokenizer loaded')
    return model, tokenizer

# Tokenize the training data
def create_train_inputs_labels(tokenizer, texts, labels):
    encoded_data = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']
    print('encoded data')

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                                        random_state=42,
                                                                                        test_size=0.2)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                           random_state=42,
                                                           test_size=0.2)
    print('created train_input and tran_masks')
    return train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks


def optimize_model_AdamW(model):
    # Define the optimizer and the loss function
    optimizer = AdamW(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    print('optimize model')
    return optimizer, loss_fn

def train(model, optimizer, loss_fn, train_inputs, train_masks, train_labels, validation_inputs, validation_masks, validation_labels, n_epochs):
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(train_inputs, attention_mask=train_masks, labels=train_labels)
        loss = loss_fn(outputs[1], train_labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(validation_inputs, attention_mask=validation_masks)
            val_loss = loss_fn(val_outputs[1], validation_labels)
            val_preds = torch.argmax(val_outputs[1], axis=1)
            val_f1 = f1_score(validation_labels, val_preds, average='macro')

        print(f'Epoch {epoch + 1}: Train Loss = {loss:.3f}, Val Loss = {val_loss:.3f}, Val F1 = {val_f1:.3f}')




###








'''

for epoch in range(3):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(train_inputs, attention_mask=train_masks, labels=train_labels)
    loss = loss_fn(outputs[1], train_labels)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(validation_inputs, attention_mask=validation_masks)
        val_loss = loss_fn(val_outputs[1], validation_labels)
        val_preds = torch.argmax(val_outputs[1], axis=1)
        val_f1 = f1_score(validation_labels, val_preds, average='macro')

    print(f'Epoch {epoch + 1}: Train Loss = {loss:.3f}, Val Loss = {val_loss:.3f}, Val F1 = {val_f1:.3f}')

'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file_csv')
    args = parser.parse_args()

    texts, labels = load_dataset(args.input_file_csv)
    model, tokenizer = load_model_tokenizer()
    train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks = create_train_inputs_labels(tokenizer, texts, labels)
    optimizer, loss_fn = optimize_model_AdamW()
    train(model, optimizer, loss_fn, train_inputs, train_masks, train_labels, validation_inputs, validation_masks,
          validation_labels, n_epochs=3)


main()



'''
# Define the model architecture
class HealthBERTClassifier(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.bert = model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Initialize the model and optimizer
model = HealthBERTClassifier()
optimizer = AdamW(model.parameters(), lr=5e-5)

# Train the model
model.train()
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(train_inputs['input_ids'], train_inputs['attention_mask'])
    loss = nn.CrossEntropyLoss()(outputs, train_labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} loss: {loss.item():.3f}")
'''