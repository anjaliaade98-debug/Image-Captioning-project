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

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------

def extract_features(image_path):

    image = load_img(image_path, target_size=(224, 224))

    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    feature = feature_extractor.predict(image, verbose=0)

    return feature


# -----------------------------
# CAPTION GENERATOR
# -----------------------------

def predict_caption(model, image_feature, tokenizer, max_length):

    in_text = "startseq"

    for i in range(max_length):

        sequence = tokenizer.texts_to_sequences([in_text])[0]

        sequence = pad_sequences(
            [sequence],
            maxlen=max_length
        )

        yhat = model.predict(
            [image_feature, sequence],
            verbose=0
        )

        yhat = np.argmax(yhat)

        word = idx_to_word(yhat, tokenizer)

        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    caption = in_text.replace("startseq", "")

    caption = caption.replace("endseq", "")

    caption = caption.strip()

    return caption

# -----------------------------
# STREAMLIT UI
# -----------------------------

st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ AI Image Caption Generator")
st.write("Upload an image and generate an AI caption.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Generate Caption"):

        with st.spinner("Generating Caption..."):

            feature = extract_features("uploaded_image.jpg")

            caption = predict_caption(
                caption_model,
                feature,
                tokenizer,
                MAX_LENGTH
            )

        st.success("Caption Generated Successfully!")

        st.subheader("Predicted Caption")

        st.write(caption)