### libraries/modules###
## Basic libraries
import pandas as pd

import numpy as np

### model loading libraries

import joblib

### UI & logic libraries

import streamlit as st

## loading trained model files 
ohe = joblib.load("ohe.pkl")
model = joblib.load("Pwr_rfr.pkl")

### UI & logic code
st.header("power generation estimation....")
st.write("This app is built to estimate solar power generation")

df=pd.read_csv("Solar power generation.csv")
st.write("Power Genertion Table")
st.dataframe(df.head(5))

st.subheader("Enter the values to estimate the power generation")

col1,col2,col3,col4 = st.columns(4)

with col1:
    SOURCE_KEY = st.selectbox("source_key :",df.SOURCE_KEY.unique())
with col2:
    DC_POWER = st.number_input("DC_Power :")
with col3:
    AC_POWER = st.number_input("AC_Power")
with col4:
    DAILY_YIELD= st.number_input("Daily_yield :")
col5,col6,col7,col8 = st.columns(4)
with col5:
    AMBIENT_TEMPERATURE = st.number_input("Ambient_temp :")
with col6:
    MODULE_TEMPERATURE  = st.number_input("Module_temp")
with col7:
    IRRADIATION = st.number_input("Irradiation")
with col8:
    TIME = st.selectbox("Time :",df.TIME.unique())

##################### Logic code#########################
if st.button("estimate"):
    row = pd.DataFrame([[SOURCE_KEY, DC_POWER, AC_POWER, DAILY_YIELD, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION, TIME]], columns=df.columns)
    

    st.write("Given Input Data:")
    
    st.dataframe(row)
    
    
    
    # Onehot Encoding
    ohedata = ohe.transform([["SOURCE_KEY","TIME"]]).toarray()
    ohedata = pd.DataFrame(ohedata, columns=ohe.get_feature_names_out())

    row=row.drop('SOURCE_KEY',axis=1)
    row=row.drop('TIME',axis=1)
    row = pd.concat([row, ohedata], axis=1)
    
    prediction = round(model.predict(row)[0])
    
    st.write(f"Estimated Power Genertion Value: {prediction}")



