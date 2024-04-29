import streamlit as st
from utils import *

st.title('Model Selection')

option = st.selectbox(
        'Select your favorite model:',
        ('--', 'RoBERTa', 'GPT-2'))

sequence = st.text_input("Enter your prompt")
generate_btn = st.button("Generate")

if len(sequence) > 0 and generate_btn:
    if option == "RoBERTa":
        st.text(generate_text_roberta(sequence))
    elif option == 'GPT-2':
        st.text(generate_text_gpt_2(sequence))