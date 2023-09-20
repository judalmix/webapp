#imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.linear_model import LinearRegression

def convert_to_string(dataframe):
    num_initial_cols = 7
    # Get the number of columns that are integers
    num_int_cols = dataframe.shape[1] - num_initial_cols
    # Create a list of new column names that are strings
    new_col_names = [str(i+num_initial_cols) for i in range(num_int_cols)]
    # Rename the integer columns using the new column names
    dataframe.columns.values[num_initial_cols:num_initial_cols + num_int_cols] = new_col_names
    for i in range(len(new_col_names)):
        df=dataframe.rename(columns={num_initial_cols:new_col_names[0]})
        num_initial_cols=num_initial_cols+1
    return df

def mirar_no_zeros(df,array,cols_totals):
    for j in range(df.shape[1]-1,6,-1):
        cols_totals=cols_totals-1
        for i in range(len(df)):
            if (df.iloc[i][j]>0 and cols_totals>0) or (df.iloc[i][j]>0 and cols_totals==0):
                array[cols_totals].append(df.iloc[i][j])

    return array

def crear_arrays(df):
    array=[]
    cols_totals=int(len(df.columns)) - 7
    for i in range(cols_totals):
        nuevo_array = []  
        array.append(nuevo_array) 
    return array, cols_totals


def truncate(number, max_decimals):
    int_part, dec_part = str(number).split(".")
    return float(".".join((int_part, dec_part[:max_decimals])))

def total_dataset(cols_totals,df,array_quartils):
    for j in range(df.shape[1]-1,6,-1):
        cols_totals=cols_totals-1
        for i in range(len(df)):
            if df.iloc[i][j]>array_quartils[cols_totals][0][2] and df.iloc[i][j]<=array_quartils[cols_totals][0][3]:
                df.iloc[i][j]=3
            elif df.iloc[i][j]>array_quartils[cols_totals][0][1] and df.iloc[i][j]<=array_quartils[cols_totals][0][2]:
                df.iloc[i][j]=2
            elif df.iloc[i][j]>array_quartils[cols_totals][0][0] and df.iloc[i][j]<=array_quartils[cols_totals][0][1]:
                df.iloc[i][j]=1
            else: 
                df.iloc[i][j]=0
    return df

def get_quartils(previa,array_quartils,df):
    quartils=[]
    for j in range(len(previa)):
        for i in range(len(array_quartils)):
            if str(i)==str(previa[0][j]):
                quartils.append(array_quartils[previa[0][j]])
    return quartils


def model(df_regression,regression_model_reduced,values_dict,num):
    num=str(num)
    target=df_regression.iloc[:,7:]
    not_target=df_regression.iloc[:,:7]
    model_fit=regression_model_reduced.fit(not_target, target)
    prediction=model_fit.predict(not_target)
    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            if prediction[i][j]<=0:
                prediction[i][j]=0
            else:
                prediction[i][j]=truncate(prediction[i][j],0)

    new_future_sales=pd.DataFrame(prediction)
    not_target = not_target.reset_index(drop=True)
    result = pd.concat([not_target, new_future_sales],axis=1)
    new_cols = {}
    for i, col in enumerate(result.columns[7:]):
        new_col_name = f'{i+1}{"  Predicció" if i%10==0 and i!=10 else ""} dels següents' + num + 'mesos'
        new_cols[col] = new_col_name 
    df_encoded = result.rename(columns=new_cols)
    df_encoded_0=df_encoded.copy()

    decodings = {}
    
    for col in df_encoded.columns[:7]:
        if col in values_dict:
            categories, _ = values_dict[col]
            decodings[col] = dict(enumerate(_))
            df_encoded[col] = df_encoded[col].map(lambda x: decodings[col][x])

    return  df_encoded_0,df_encoded

st.write('Please have patience, some computations are running in this time!')
dataset=st.session_state.data
num=st.session_state.numero

df_not_encoded=st.session_state.data_not_encoded
dataset_grouped_reduced=st.session_state.data_reduced
values_dict=st.session_state.diccionari
previa=st.session_state.num_mes_previ


df=convert_to_string(dataset)
df_regression=convert_to_string(dataset)
df_encoded,df_not_encoded=model(df_regression, LinearRegression(),values_dict,num)


nom_ultima_col = df.columns[-1]
array,cols_totals=crear_arrays(df)
final_array=mirar_no_zeros(df,array,cols_totals)
array_quartils,cols_totals=crear_arrays(df)

for i in range(len(array)):
    cols_totals=cols_totals-1
    dataset = pd.DataFrame(final_array[i])
    array_quartils[cols_totals].append(np.quantile(dataset, [0.25,0.5,0.75,1]))


previa_per_quartils=[]
for i in range(len(previa)):
    previa_per_quartils.append(int(previa[i])-7)

previa_per_quartils=pd.DataFrame(previa_per_quartils)


df=total_dataset(cols_totals,df,array_quartils)
quartilss=get_quartils(previa_per_quartils,array_quartils,df)
quartilss=pd.DataFrame(quartilss)


def truncate(number, max_decimals):
    int_part, dec_part = str(number).split(".")
    return float(".".join((int_part, dec_part[:max_decimals])))

primer=(quartilss[0][0][0]+quartilss[0][1][0]+quartilss[0][2][0]+quartilss[0][3][0])/4
segon=(quartilss[0][0][1]+quartilss[0][1][1]+quartilss[0][2][1]+quartilss[0][3][1])/4
tercer=(quartilss[0][0][2]+quartilss[0][1][2]+quartilss[0][2][2]+quartilss[0][3][2])/4
quart=(quartilss[0][0][3]+quartilss[0][1][3]+quartilss[0][2][3]+quartilss[0][3][3])/4

primer=truncate(primer,0)
segon=truncate(segon,0)
tercer=truncate(tercer,0)
quart=truncate(quart,0)

quartilss=[primer,segon,tercer,quart]

if 'quartils' not in st.session_state:
    st.session_state.quartils=quartilss
if 'df' not in st.session_state:
    st.session_state.df=df
if 'array_quartils' not in st.session_state:
    st.session_state.array_quartils=array_quartils
if 'df_regression' not in st.session_state:
    st.session_state.df_regression=df_regression
if 'previa_varis' not in st.session_state:
    st.session_state.previa_varis=previa_per_quartils
if 'df_model' not in st.session_state:
    st.session_state.df_model=df_not_encoded


st.write('You can go to the next page!'+':innocent:')


