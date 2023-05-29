#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "request pretrained BERT model"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2023 by Eva Seidlmayer"
__license__ = "ISC license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1 "


import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('models/bert-base_t10k_e3_lr3e-5')

# Preprocess the specific text
text = "Your specific text here"
preprocessed_text = text  # Implement this function to preprocess the text

# Convert the preprocessed text into a format compatible with the model
input_data = np.array([preprocessed_text])

# Feed the text to the model for prediction
predictions = model.predict(input_data)

# Obtain the classification result
predicted_class = np.argmax(predictions, axis=1)
