import streamlit as st
import matplotlib.pyplot as plt
import pickle 
import os 
import sys
import model 
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

with open("my_model.pkl", "rb") as f:
    w, b = pickle.load(f)
with open("my_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

st.title('🚨📩 Email Spam Detection ')

user_text = st.text_input(label="Enter Email Text: ")
if user_text is not None:
    
    st.markdown(
    """
    <style>
    div.stButton > button {
        height: 60px;
        width: 200px;
        font-size: 20px;
        margin-left:220px;
        margin-down:40px;
        font-family: Arial, Helvetica, sans-serif;
        font-size: 200px;
        font-color: red;
    }
    
    </style>
    """,
    unsafe_allow_html=True)
    
    st.markdown(
    """
    <style>
    /* Change font globally */
    html, body, [class*="css"] {
        font-family: 'Arial', sans-serif;
        font-size: 20px;
        color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True)
    
    
    st.write('')
    st.write("\n\nEntered Email Text : ",user_text)
    st.write('')
    submit = st.button("Submit",property)
    if submit:
        
        st.write('✅ Submitted')
        cleaned_text = model.preprocessed_text(user_text)
        vectorized_input = model.tfidf_vectorizer.transform([cleaned_text]).toarray()
        prediction = model.logistic_regression(vectorized_input,w=w,b=b)
        pred = ['Spam' if prediction > 0.5 else 'Not Spam']
        st.write("📋 Prediction Results : ", prediction)
        st.write('Result : ',pred[0])

        spam_percentage = np.mean(prediction) * 100
        not_spam_percentage = (1 - np.mean(prediction)) * 100
        
        
        # prediction_series = pd.Series(prediction)
        # labels = prediction_series.apply(lambda x: 'Spam' if x > 0.5 else 'Not Spam')
        # value_counts = labels.value_counts()
        
        # Visualization
        # Pie chart
        st.subheader("🔍 Real vs Spam Distribution")
        fig1, ax1 = plt.subplots()
        ax1.pie([spam_percentage, not_spam_percentage],
                labels=['Spam','Not Spam'],
                autopct='%1.1f%%',
                colors=['red','green']
                )
        st.pyplot(fig1)
