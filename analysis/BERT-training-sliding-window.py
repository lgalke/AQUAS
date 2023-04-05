#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "train datatset with BERT using sliding windows approach"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2023 by Eva Seidlmayer"
__license__ = "ISC license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1 "

import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification, BertForPreTraining
import numpy as np
import tensorflow as tf
import argparse
import os
from sklearn.metrics import f1_score
import torch



def load_dataset(input_file_csv):
    # Load dataset
    df = pd.read_csv(input_file_csv, sep=',')
    df = df.sample(frac=1)
    df = df.astype(str)
    texts = df['text'].to_list()
    labels = df['category-id'].to_list()
    print('data input lists created')
    return texts, labels


def tokenize(texts):
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # set max_length
    max_length = 100000

    # Tokenize the text data
    tokens = tokenizer(texts, max_length=max_length, padding='max_length', truncation=True, return_tensors="tf")
    print('text is tokenized')
    return tokens


def convert_labels(labels):
    # Convert labels to numerical values
    label_map = {'1': 0, '2': 1, '3': 2}
    labels = [label_map[label] for label in labels]
    labels_conv = tf.keras.utils.to_categorical(labels, num_classes=3)
    print('labels converted')
    return labels_conv


def calc_split_ratio(labels_conv):
    # 80% training 20% validation
    split_ratio = int(len(labels_conv) * 0.2)
    print('split_ratio defined')
    return split_ratio

def split_train_val_data(tokens, split_ratio, labels_conv):
    # Split the data into training and validation sets
    train_inputs, val_inputs = np.split(tokens['input_ids'], [split_ratio])
    train_masks, val_masks = np.split(tokens['attention_mask'], [split_ratio])
    train_labels, val_labels = np.split(labels_conv, [split_ratio])
    print('train/val -inputs, -masks, -labels created')
    return train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels


def sliding_window(item):
    window_size = 512
    stride = 256
    #elem = item.split()
    windows = [item[i:i+window_size] for i in range(0, len(item)-window_size+1, stride)]
    return windows


def fine_tune_BERT(train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels):
    # Fine-tune pre-trained BERT model
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    model.fit([train_inputs, train_masks], train_labels, validation_data=([val_inputs, val_masks], val_labels), epochs=3, batch_size=8)
    print('BERT fine tuned')
    return model


def evaluate_model(model, val_inputs, val_masks, val_labels):
    # Evaluate  model
    results = model.evaluate([val_inputs, val_masks], val_labels, batch_size=8)

    output = model.predict([val_inputs, val_masks])
    predicted_labels = output.logits.argmax(axis=1)
    f1 = f1_score(val_labels.argmax(axis=1), predicted_labels, average='weighted')

    print("Validation Loss: {:.4f} Accuracy: {:.4f} F1-score: {:.4f}".format(*results, f1))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file_csv')
    args = parser.parse_args()


    texts, labels = load_dataset(args.input_file_csv)
    tokens = tokenize(texts)
    labels_conv = convert_labels(labels)
    split_ratio = calc_split_ratio(labels_conv)
    train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels= split_train_val_data(tokens, split_ratio, labels_conv)

    for item in train_inputs:
        if len(item) > 512:
            windows = sliding_window(item)
            for window in windows:
                fine_tune_BERT(window)

        else:
            fine_tune_BERT(item)


    #model = fine_tune_BERT(train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels)
    #evaluate_model(model, val_inputs, val_masks, val_labels)


main()