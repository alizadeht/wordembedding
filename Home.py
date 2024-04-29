import streamlit as st

# Set the page configuration for the homepage
st.set_page_config(page_title="Contextualized Word Embeddings Model App", layout="wide")

# Main header of the homepage
st.title("Welcome to our Contextualized Word Embeddings Model App!")

col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.image("/Users/turalalizada/Desktop/SDP II/az-word-embedding-main 2/Screenshot 2023-12-24 at 21.46.32.png", caption="Biz yaradaq, siz yazın!")


# Introduction or About section
st.header("About our app")
st.write("""
         This is our page of Contextualized Word Embeddings Model. 
         We as a team, trained 2 models, RoBERTo, and GPT-2 in order to achieve our goal. If you want to test our model, go to the Model Selection page, choose one of these models, type anything you want, and run. 
         
         """)
st.write("""
         P.S: As RoBERTo model is mask language model, you have to write your text like this:
         """)
st.write("""
         Tural <mask> gedirdi.
         """)






# Footer
st.write("---")
st.write("© 2023 azerb(AI)jan app - All rights reserved")
