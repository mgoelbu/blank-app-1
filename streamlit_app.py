import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

st.title("GPT-2 Text Generator")

# User input for the prompt
prompt = st.text_input("Enter your prompt:", value="The future of AI is")

# User input for the number of tokens
num_tokens = st.number_input("Enter number of tokens to generate:", min_value=5, max_value=100, value=20)

# Generate text with different creativity levels
if st.button("Generate Response"):
    # Generate text with high creativity (lower predictability, temperature > 1.0)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs_high = model.generate(inputs['input_ids'], max_length=num_tokens, do_sample=True, temperature=1.5)
    generated_text_high = tokenizer.decode(outputs_high[0], skip_special_tokens=True)
    st.write("High Creativity Response:")
    st.write(generated_text_high)

    # Generate text with low creativity (higher predictability, temperature < 1.0)
    outputs_low = model.generate(inputs['input_ids'], max_length=num_tokens, do_sample=True, temperature=0.7)
    generated_text_low = tokenizer.decode(outputs_low[0], skip_special_tokens=True)
    st.write("Low Creativity Response:")
    st.write(generated_text_low)
