#imports

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO



#functions
# Define a function to read CSV data

def load_csv(file):
    df = pd.read_csv(file, sep=";")
    return df


st.set_page_config(page_title= 'Zippers prediction')
st.title("Kreband: Zippers prediction")

st.write(' ')
st.write(' ')

st.write('This is a WEBAPP created to analyze and predict thezippers will be more sold in the future months of the year. In this webs app you will be able to two types of predictions. Firstly, you will be able to predict for one zipper of the data set the quantity of zippers of this type that will be sold in the next months. The second one, you will be able to do the prediction for all the data for the next months. ')
st.write('')
st.write('Before starting with the predictions, you must follow the steps to navigate through this website. ')
st.write('1-Import the data')
st.write('2-Go to the page ‘Data Distribution’ to group the columns by the quantity of months you want to do the predictions.')
st.write('3-Go to the page ‘Zipper prediction’ to see the prediction.')
st.write('')
st.write('')
st.write("Let's start! ")
st.write('')


st.sidebar.info('Select a page above')
st.header("Data visualitzation")
st.write('')
#uploading the dataset
st.write('Here you can upload your data: ')
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None: 
    df=load_csv(uploaded_file)
    st.write('')
    df['Llargada'] = df['Llargada'].astype(str)
    df['Sliders'] = df['Sliders'].astype(str)
    for i in range(len(df)):
        if len(df['Llargada'].iloc[i])<3:
            df['Llargada'].iloc[i]='0'+df['Llargada'].iloc[i]
        else:
            df['Llargada'].iloc[i]=df['Llargada'].iloc[i]
    for i in range(len(df)):
        if len(df['Sliders'].iloc[i])<3:
            df['Sliders'].iloc[i]='0'+df['Sliders'].iloc[i]
        else:
            df['Sliders'].iloc[i]=df['Sliders'].iloc[i]

    df['Llargada'] = df['Llargada'].astype(object)
    df['Sliders'] = df['Sliders'].astype(object)
    st.write(df)
    st.write('Our Data has: ', df.shape[0], 'rows and', df.shape[1],'columns' )
    st.write('Please, now go to the Data Distribution page to see how the data is distributed. ')

    if "dataframe45" not in st.session_state:
        st.session_state["dataframe45"] = df
    