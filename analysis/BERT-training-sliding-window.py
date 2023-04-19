#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "train datatset with BERT using sliding windows approach"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2023 by Eva Seidlmayer"
__license__ = "ISC license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1 "

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, BertForPreTraining
import numpy as np
import tensorflow as tf
import argparse
import os
from sklearn.metrics import f1_score
import torch
from typing import Optional, Union, Tuple
from transformers.modeling_outputs import SequenceClassifierOutput

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


# used in class AQUAS_slidingwindow
def sliding_window(item):
    window_size = 512
    stride = 256
    #elem = item.split()
    windows = [item[i:i+window_size] for i in range(0, len(item)-window_size+1, stride)]
    return windows

class AQUAS_slidingwindow(BertForSequenceClassification):
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        AQUASwindowsvectors = []
        AQUASnumberwindows = 0
        for item in input_ids:
            if len(item) > 512:
                windows = sliding_window(item)
                for window in windows:
                    #finetune_windw = fine_tune_BERT(window)
                    #vector.append(finetune_windw)

                    # replaced "self" by "window"
                    outputs = window.Sbert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )
                # get the vector
                    pooled_output = outputs[1]

                    AQUASwindowsvectors.append(pooled_output)
                    AQUASnumberwindows += 1
                #sum and mean
                #AQUASpooled_output = [sum(i) for i in zip(*AQUASwindowsvectors)] / AQUASnumberwindows
                AQUASpooled_output = tf.reduce_sum(AQUASwindowsvectors, 0)
                AQUASpooled_output = tf.divide(AQUASpooled_output/AQUASnumberwindows)
                return AQUASpooled_output
            else:
                #replaced "self" by "item"
                outputs = item.AQUASbert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                AQUASpooled_output = outputs[1]
                return AQUASpooled_output

        pooled_output = self.dropout(AQUASpooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    print('sliding window: check!')


#
'''
def fine_tune_BERT(train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels):
    # Fine-tune pre-trained BERT model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
'''


def fit_model(train_inputs_prep, val_inputs_prep,train_masks, val_masks, train_labels, val_labels):
    AQUASbert = AQUAS_slidingwindow.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    model.AQUASbert([train_inputs_prep, train_masks], train_labels, validation_data=([val_inputs_prep, val_masks], val_labels), epochs=3, batch_size=8)
    #print('BERT fine tuned')
    print('model trained')
    return model


def evaluate_model(model, val_inputs, val_masks, val_labels):
    # Evaluate  model
    results = model.evaluate([val_inputs, val_masks], val_labels, batch_size=8)
    output = model.predict([val_inputs, val_masks])
    predicted_labels = output.logits.argmax(axis=1)
    f1 = f1_score(val_labels.argmax(axis=1), predicted_labels, average='weighted')
    print('Validation Loss: {:.4f} Accuracy: {:.4f} F1-score: {:.4f}'.format(*results, f1))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file_csv')
    args = parser.parse_args()

    texts, labels = load_dataset(args.input_file_csv)
    tokens = tokenize(texts)
    labels_conv = convert_labels(labels)
    split_ratio = calc_split_ratio(labels_conv)
    train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels= split_train_val_data(tokens, split_ratio, labels_conv)

    train_inputs_prep = AQUAS_slidingwindow(train_inputs)
    val_inputs_prep = AQUAS_slidingwindow(val_inputs)
    model = fit_model(train_inputs_prep, val_inputs_prep, train_masks, val_masks, train_labels, val_labels)
    evaluate_model(model, val_inputs, val_masks, val_labels)


main()