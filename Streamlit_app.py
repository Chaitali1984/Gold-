import pandas as pd
import streamlit as st 
from statsmodels.tsa.arima_model import ARIMA
from pickle import dump
from pickle import load

st.title('Model Deployment: ARIMA')

st.sidebar.header('User Input Parameters')

def user_input_features():
   price =  st.sidebar.number_input("30")

   data = {'price'}


   df = user_input_features()
   st.subheader('User Input parameters')
   st.write(df)

   Gold_Price  = pd.read_csv("C:/Users/Hp/Downloads/Gold_data.csv")

   train = Gold_Price [:2182]
   test  = Gold_Price [2182:]
   
   clf = ARIMA()
   clf.fit(1,2182)

  
prediction_proba = clf.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

st.subheader('Prediction Probability')
st.write(prediction_proba)
   
