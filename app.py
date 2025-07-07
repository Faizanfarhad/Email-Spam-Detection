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
        font-size: 20px;
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
    if submit and user_text != '':
        
        st.write('✅ Submitted')
        try:
            cleaned_text = model.preprocessed_text(user_text)

            vectorized_input = tfidf_vectorizer.transform([cleaned_text]).toarray()

            prediction = model.logistic_regression(vectorized_input,w=w,b=b)

            pred = ['Spam' if prediction > 0.5 else 'Not Spam']

            color = 'red' if pred[0] == 'Spam' else 'green'
            result = pred[0]
            pred_result ='📋 Prediction Results :'
            st.markdown(f' <span style="color:{color}; font-size: 40px; font-weight:bold;">{pred_result}{result}</span>', unsafe_allow_html=True)
            
            spam_percentage = np.mean(prediction) * 100

            not_spam_percentage = (1 - np.mean(prediction)) * 100
        

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
            st.write()
            st.write()
            st.header('📊 Model Scores')
            st.subheader("🎯 Accuracy Score : 0.9526570048309179 (95%) ")
            st.subheader("🔍 Precision Score  : 0.9265734265734266 (92%) ")
            st.subheader("📊 Recall Score : 0.9044368600682594 (90%) ")
            st.subheader("⚖️ F1 Score  : 0.9153713298791019 (91%) ")
        except Exception as e:
            st.error(f'Something went wrong {e}')
if submit and user_text.strip() == "":
    st.warning("⚠️ Please enter some text before submitting.")
    
