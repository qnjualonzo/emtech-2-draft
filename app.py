import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import torch
import numpy as np

# Load Model and Tokenizer
MODEL_REPO = "google/mt5-small"
st.title("Multilingual Translation with MT5")

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_REPO)
    model = model.cuda() if torch.cuda.is_available() else model
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Language Token Mapping
LANG_TOKEN_MAPPING = {
    'English': '<en>',
    'Filipino': '<fil>',
    'Japanese': '<ja>'
}

# Utility Functions
def encode_input(text, target_lang, tokenizer, seq_len=128):
    task_prefix = f"translate English to {target_lang}: "
    input_text = task_prefix + text
    input_ids = tokenizer.encode(
        input_text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=seq_len
    )
    return input_ids

@st.cache_resource
def load_dataset_cache():
    return load_dataset("alt")

dataset = load_dataset_cache()

# UI Components
st.sidebar.header("Settings")
source_language = st.sidebar.selectbox("Source Language", options=LANG_TOKEN_MAPPING.keys())
target_language = st.sidebar.selectbox("Target Language", options=LANG_TOKEN_MAPPING.keys())

st.header("Translation Input")
input_text = st.text_area("Enter text to translate", value="", height=150)

if st.button("Translate"):
    if source_language == target_language:
        st.error("Source and target languages must be different.")
    elif not input_text.strip():
        st.error("Please enter text to translate.")
    else:
        with st.spinner("Translating..."):
            input_ids = encode_input(input_text, target_language, tokenizer)
            input_ids = input_ids.cuda() if torch.cuda.is_available() else input_ids

            model_out = model.generate(input_ids)
            output_text = tokenizer.decode(model_out[0], skip_special_tokens=True)

        st.success("Translation Complete!")
        st.subheader("Translated Text")
        st.text(output_text)

st.sidebar.header("Dataset Exploration")
if st.sidebar.checkbox("Show Dataset Example"):
    st.subheader("Dataset Example")
    example = dataset['train'][0]['translation']
    st.json(example)

# Streamlit App Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed with ❤️ using [Streamlit](https://streamlit.io/) and Hugging Face Transformers.")
