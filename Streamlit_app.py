

import streamlit as st
import datetime



#Importing Libreries

import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing

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
