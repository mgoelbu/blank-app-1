import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import random


my_secret_key = st.secrets['MyOpenAIKey']

BUID = 45664861  # Replace with your actual BU ID
random.seed(BUID)
torch.manual_seed(BUID)

# Load GPT-2 model and tokenizer from Hugging Face
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

st.title("GPT-2 Text Generator")

# User input for the prompt
prompt = st.text_input("Enter your prompt:", value="I love studying Deploying Generative AI in the Enterprise")


num_tokens = st.number_input("Enter number of tokens to generate:", min_value=10, max_value=100, value=50)


if st.button("Generate Response"):
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate text with high creativity (temperature=1.5)
    outputs_high = model.generate(
        inputs['input_ids'], 
        max_length=num_tokens, 
        do_sample=True, 
        temperature=1.5
    )
    generated_text_high = tokenizer.decode(outputs_high[0], skip_special_tokens=True)

    # Generate text with low creativity (temperature=0.7)
    outputs_low = model.generate(
        inputs['input_ids'], 
        max_length=num_tokens, 
        do_sample=True, 
        temperature=0.7
    )
    generated_text_low = tokenizer.decode(outputs_low[0], skip_special_tokens=True)

    # Display High Creativity Response with markdown formatting
    st.markdown("### **High Creativity Response:**")
    st.markdown(f"> {generated_text_high}")  # Quoting the text for markdown

    # Divider between high and low creativity responses
    st.markdown("---")

    # Display Low Creativity Response with markdown formatting
    st.markdown("### **Low Creativity Response:**")
    st.markdown(f"> {generated_text_low}")  # Quoting the text for markdown
