import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os


my_secret_key = st.secrets['MyOpenAIKey'] 


BUID = 45664861  # Replace with your actual BU ID
torch.manual_seed(BUID)

# Load GPT-2 model and tokenizer from Hugging Face
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

st.title("GPT-2 Text Generator")

# User input for the prompt
prompt = st.text_input("Enter your prompt:", value="The future of AI is")

# User input for the number of tokens
num_tokens = st.number_input("Enter number of tokens to generate:", min_value=10, max_value=100, value=50)

# Generate text with different levels of creativity
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
    st.write("High Creativity Response:")
    st.write(generated_text_high)

    # Generate text with low creativity (temperature=0.7)
    outputs_low = model.generate(
        inputs['input_ids'], 
        max_length=num_tokens, 
        do_sample=True, 
        temperature=0.7
    )
    generated_text_low = tokenizer.decode(outputs_low[0], skip_special_tokens=True)
    st.write("Low Creativity Response:")
    st.write(generated_text_low)
