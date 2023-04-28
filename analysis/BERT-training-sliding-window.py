#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "train datatset with BERT using sliding windows approach"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2023 by Eva Seidlmayer"
__license__ = "ISC license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1 "

# BERT_MODEL_IDENTIFIER = "bert-base-uncased"
BERT_MODEL_IDENTIFIER = "dmis-lab/biobert-v1.1"

import pandas as pd
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertForPreTraining,
)
import numpy as np
import argparse
import os
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from transformers.modeling_outputs import SequenceClassifierOutput


try:
    import wandb
except ImportError:
    print("Wandb not installed. Not using wandb. To use: pip install wandb")


def load_dataset(input_file_csv):
    # Load dataset
    df = pd.read_csv(input_file_csv, sep=",")
    df = df.sample(frac=1)
    df = df.astype(str)
    texts = df["text"].to_list()
    labels = df["category-id"].to_list()
    print("data input lists created")
    return texts, labels


def tokenize(texts):
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_IDENTIFIER)

    # set max_length
    max_length = 2048

    # Tokenize the text data
    tokens = tokenizer(
        texts,max_length =max_length, padding="max_length", truncation=True
    )
    print("text is tokenized")
    return tokens


def convert_labels(labels):
    # Convert labels to numerical values
    label_map = {"1": 0, "2": 1, "3": 2}
    labels = [label_map[label] for label in labels]
    labels_conv = torch.tensor(labels)

    print("labels converted")
    return labels_conv


def calc_split_ratio(labels_conv):
    # 80% training 20% validation
    split_ratio = int(len(labels_conv) * 0.2)
    print("split_ratio defined")
    return split_ratio


def split_train_val_data(tokens, split_ratio, labels_conv):
    # Split the data into training and validation sets
    train_inputs, val_inputs = np.split(tokens["input_ids"], [split_ratio])
    train_masks, val_masks = np.split(tokens["attention_mask"], [split_ratio])
    train_labels, val_labels = np.split(labels_conv, [split_ratio])
    print("train/val -inputs, -masks, -labels created")
    return train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels


# used in class AQUAS_slidingwindow
def sliding_window(item):
    window_size = 512
    stride = 256
    windows = [
        item[i : i + window_size] for i in range(0, len(item) - window_size + 1, stride)
    ]
    return windows


class AQUASSlidingBERT(BertForSequenceClassification):
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

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        AQUASwindowsvectors = []
        AQUASnumberwindows = 0

        # input_ids : [bsz, max_seqlen]
        # attention_mask: [bsz, max_seqlen]
        # 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0

        batch_size = input_ids.size(0)

        assert batch_size == 1, "Please use batch size = 1"

        length = attention_mask.sum(1)

        if length > 512:
            print("Len > 512, sliding")
            window_tokens = sliding_window(item)
            window_attn_masks = sliding_window(attention_mask)

            for tokens, attn_mask in zip(window_tokens, window_attn_masks):
                tokens = tokens.unsqueeze(0)
                attn_mask = attn_mask.unsqueeze(0)
                print("\tTokens size", tokens.size())
                print("\tattn_mask size", attn_mask.size())
                print("\tposition_ids", position_ids)

                outputs = self.bert(
                    tokens,
                    attention_mask=attn_mask,
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
            # sum and mean
            # AQUASpooled_output = [sum(i) for i in AQUASwindowsvectors] / AQUASnumberwindows
            AQUASpooled_output = torch.stack(AQUASwindowsvectors, dim=0).mean(dim=0)

        else:
            print("Len <= 512, no slides :(")
            input_ids = input_ids.unsqueeze(0)
            print("\tInput_ids size", input_ids.size())
            print("\tattention_mask size", attention_mask.size())
            print("\tposition_ids", position_ids)
            outputs = self.bert(
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

        # AQUASpooled_output defined
        pooled_output = self.dropout(AQUASpooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
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
        print("sliding window: check!")


def train_epoch(model, optimizer, train_inputs, train_labels, train_masks):
    # optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0)

    # bisher: batch size 1, mehr spaeter

    train_loader = torch.utils.data.DataLoader(
        list(zip(train_inputs, train_labels, train_masks)), batch_size=1, shuffle=True
    )

    # training for one epoch
    for batch in train_loader:
        optimizer.zero_grad()

        # batch auseinanerfriemeln
        batch_inputs, batch_labels, batch_masks = batch
        output = model(
            input_ids=batch_inputs, labels=batch_labels, attention_mask=batch_masks
        )

        # output: SequenceClassifierOutput
        loss = output["loss"]  # oder output.loss

        loss.backward()
        optimizer.step()

    # model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    print("model trained")
    return model


def evaluate_model(model, val_inputs, val_masks, val_labels):
    # Evaluate  model
    val_loader = torch.utils.data.DataLoader(
        list(zip(val_inputs, val_masks)),
        batch_size=1,
        shuffle=False,  # Never change to True, else all will break
    )

    predictions = []
    with torch.no_grad():
        model.eval()
        for batch_input, batch_mask in val_loader:
            outputs = model(input_ids=batch_input, attention_mask=val_masks)
            logits = outputs[1]
            assert logits.size(1) == 3, "Something went terribly wrong"
            predicted_class = torch.argmax(logits, dim=1)

            predictions.append(predicted_class)

    predictions = torch.stack(predictions)
    # Sliding
    # output = model.predict([val_inputs, val_masks])
    # predicted_labels = output.logits.argmax(axis=1)

    accuracy = (predictions == val_labels).mean()

    f1 = f1_score(val_labels, predictions, average="weighted")

    return accuracy, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_csv")
    args = parser.parse_args()

    learning_rate = 5e-5
    epochs = 10

    run = wandb.init(
        # Set the project where this run will be logged
        project="AQUAS",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "epochs": epochs,
        },
    )

    texts, labels = load_dataset(args.input_file_csv)
    tokens = tokenize(texts)
    labels_conv = convert_labels(labels)
    split_ratio = calc_split_ratio(labels_conv)
    (
        train_inputs,
        val_inputs,
        train_masks,
        val_masks,
        train_labels,
        val_labels,
    ) = split_train_val_data(tokens, split_ratio, labels_conv)

    # OUR AQUASBert INIT
    model = AQUASSlidingBERT.from_pretrained(
        BERT_MODEL_IDENTIFIER, num_labels=3
    )  # BioBERT statt bert-base-uncased
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    wandb.watch(model)
    print("weight and biases is tracking")

    # each loop is one epoch
    for epoch in range(epochs):
        train_epoch(model, optimizer, train_inputs, train_labels, train_masks)
        acc, f1 = evaluate_model(model, val_inputs, val_masks, val_labels)

        wandb.log({"accuracy": acc, "f1": f1})

        print(f"[{epoch+1}] Accuracy: {acc:.4f} F1-score: {f1:.4f}")


if __name__ == "__main__":
    main()
