

import streamlit as st
import datetime



#Importing Libreries

import pandas as pd



import warnings
df=pd.read_csv('Gold_data.csv')

hwmodel=ExponentialSmoothing(df['price'],seasonal='mul',trend='add',seasonal_periods=24).fit()



def main():
    st.title("Gold Price Predictor")
    st.info("Let us predict the Price of GOLD for the Future")

    s = datetime.date(2022,5,24)
    e = st.date_input("Enter the ending Date to Predict the Gold Prices")
    diff=( (e-s).days+1)
      
    if st.button("PREDICT"):
        index_future_dates=pd.date_range(start= s ,end= e)
        pred=hwmModel.forecast(diff).rename('Price')
        pred.index=index_future_dates
        df = pd.DataFrame(pred)

        st.dataframe(df)

        st.line_chart(df)








if __name__ == '__main__':
   main() 
import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load

st.title('Model Deployment: Logistic Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    CLMSEX = st.sidebar.selectbox('Gender',('1','0'))
    CLMINSUR = st.sidebar.selectbox('Insurance',('1','0'))
    SEATBELT = st.sidebar.selectbox('SeatBelt',('1','0'))
    CLMAGE = st.sidebar.number_input("Insert the Age")
    LOSS = st.sidebar.number_input("Insert Loss")
    data = {'CLMSEX':CLMSEX,
            'CLMINSUR':CLMINSUR,
            'SEATBELT':SEATBELT,
            'CLMAGE':CLMAGE,
            'LOSS':LOSS}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


# load the model from disk
loaded_model = load(open('Logistic_Model.sav', 'rb'))

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

st.subheader('Prediction Probability')
st.write(prediction_proba)



