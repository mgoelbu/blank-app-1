import streamlit as st
import openai
import random
import numpy as np
import os

my_secret_key = st.secrets['MyOpenAIKey']
os.environ["OPENAI_API_KEY"] = my_secret_key
openai.api_key = os.getenv("OPENAI_API_KEY")


BUID = 45664861 # Replace with your actual BU ID (numeric part only)

# Seed the random generators for consistency
random.seed(BUID)
np.random.seed(BUID)

st.title("GPT-2 Text Generator using OpenAI")

# User prompt input
prompt = st.text_input("Enter your prompt:", value="Barcelona is a")

# Number of tokens input
num_tokens = st.number_input("Enter number of tokens to generate:", min_value=10, max_value=100, value=50)

# Generate the text when the button is pressed
if st.button("Generate Response"):
    # High creativity (temperature = 1.5)
    response_high = openai.Completion.create(
        engine="text-davinci-002",  # You can switch to another engine if needed
        prompt=prompt,
        max_tokens=int(num_tokens),
        temperature=1.5  # High creativity
    )
    st.write("High Creativity Response:")
    st.write(response_high.choices[0].text.strip())

    # Low creativity (temperature = 0.7)
    response_low = openai.Completion.create(
        engine="text-davinci-002",  # Use the same engine
        prompt=prompt,
        max_tokens=int(num_tokens),
        temperature=0.7  # Low creativity
    )
    st.write("Low Creativity Response:")
    st.write(response_low.choices[0].text.strip())
