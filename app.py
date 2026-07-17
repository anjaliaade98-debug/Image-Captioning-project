import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

# -----------------------------
# CONFIG
# -----------------------------

MAX_LENGTH = 34

# -----------------------------
# LOAD TOKENIZER
# -----------------------------

with open("tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

# -----------------------------
# LOAD CAPTION MODEL
# -----------------------------

caption_model = load_model("best_model.keras")

# -----------------------------
# LOAD VGG16 FEATURE EXTRACTOR
# -----------------------------

base_model = VGG16(weights="imagenet")

feature_extractor = Model(
    inputs=base_model.inputs,
    outputs=base_model.layers[-2].output
)

# -----------------------------
# IDX TO WORD
# -----------------------------

def idx_to_word(integer, tokenizer):

    for word, index in tokenizer.word_index.items():

        if index == integer:
            return word

    return None