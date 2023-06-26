import streamlit as st
import numpy as np 
import pandas as pd 
import pickle

log_model=pickle.load(open('log_model.pkl','rb'))
import joblib
vectorizer = joblib.load('vectorizer.pkl')
# svm=pickle.load(open('svm.pkl','rb'))

def main():
    
    st.title("Text Classifier")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Email Text Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    text=[]
    inp=""
    

    inp=st.text_input("Enter the email: ")
    text=[inp]
    
    
    vectorized_email = vectorizer.transform(text) 
    

    predicted_class = log_model.predict(vectorized_email)
    if st.button("Predict"):
       st.success(predicted_class[0])



if __name__=='__main__':
    main()

