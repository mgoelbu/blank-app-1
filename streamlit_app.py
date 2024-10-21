import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

st.title("GPT-2 Text Generator")

# Text input from user
prompt = st.text_input("Enter your prompt:", value="The future of AI is")

# Number of tokens input from user
num_tokens = st.number_input("Enter number of tokens to generate:", min_value=10, max_value=100, value=50)

# Generate text with different levels of creativity
if st.button("Generate Response"):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # High creativity (low predictability)
    outputs_high = model.generate(
        inputs['input_ids'], 
        max_length=num_tokens, 
        do_sample=True, 
        temperature=1.5,  # High creativity
        top_p=0.95
    )
    generated_text_high = tokenizer.decode(outputs_high[0], skip_special_tokens=True)
    st.write("High Creativity Response:")
    st.write(generated_text_high)
    
    # Low creativity (high predictability)
    outputs_low = model.generate(
        inputs['input_ids'], 
        max_length=num_tokens, 
        do_sample=True, 
        temperature=0.7,  # Low creativity
        top_p=0.95
    )
    generated_text_low = tokenizer.decode(outputs_low[0], skip_special_tokens=True)
    st.write("Low Creativity Response:")
    st.write(generated_text_low)
