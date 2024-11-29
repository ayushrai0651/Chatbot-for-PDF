import streamlit as st
from transformers import pipeline

# Load the pre-trained GPT-2 model from Hugging Face
generator = pipeline('text-generation', model='gpt2')

# Set up the Streamlit app interface
st.title("Simple Chatbot")

# Create a text input box for the user to ask questions
user_input = st.text_input("You:", "")

if user_input:
    # Generate a response using GPT-2
    response = generator(user_input, max_length=100, num_return_sequences=1)[0]['generated_text']
    
    # Display the chatbot's response
    st.write(f"Bot: {response}")
