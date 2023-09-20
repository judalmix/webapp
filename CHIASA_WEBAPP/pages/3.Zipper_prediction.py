#imports

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import pickle
import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost.sklearn import XGBClassifier

#functions

def apply_regression(data,previa):
    model=LinearRegression()
    y=data.loc[: , previa]
    x=data.loc[: , data.drop(previa, axis=1).columns]
    X_train, X_test, Y_train, Y_test= train_test_split(x,y,train_size=0.8)

    model_fit=model.fit(X_train, Y_train)
    predict = model_fit.predict(X_test)

    relative_error=np.abs(predict-Y_test)/np.abs(Y_test)
    relative_error_mean=relative_error.mean()

    return model, X_test, x, Y_test,relative_error_mean,predict


def multiclass_classification(data,previa):
    model_xgboost=XGBClassifier()
    #st.write(data1.loc[0])
    #important=data1.loc[:,data1.drop(previa, axis=1).columns]

    column_names = data.columns.tolist()

    # Create a new list of column names starting from column 7
    new_column_names = [i if i >= 7 else column_names[i] for i in range(len(column_names))]
    # Assign the new column names to the DataFrame
    data.columns = new_column_names 
    data1=data.copy()
    st.write(data1)
    for i in data1.columns:
        if i==int(previa):
            st.write('YES')
        elif i==previa:
            st.write('YESS 2')
        elif i==str(previa):
            st.write('YESS 3')
    y2=data.loc[: , int(previa)]
    st.write(y2)
    x2=data.loc[: , data.drop(int(previa), axis=1).columns]
    X_train2, X_test2, Y_train2, Y_test2= train_test_split(x2,y2,train_size=0.8, test_size=0.2)

    model_xgboost_multiclass = model_xgboost.fit(X_train2, Y_train2)
    predict_xgb = model_xgboost_multiclass.predict(X_test2)

    return  predict_xgb,model_xgboost,X_train2, X_test2, Y_train2, Y_test2,x2,y2



def array_to_dataset(idx,num_cols,new_df):
    new_data=pd.DataFrame()
    j=0
    for i in new_df.columns: 
        if j<num_cols:
            new_data[i]=[idx[j]]
            new_data.i=new_data.astype(int)
        j=j+1
    return new_data


def insert_zipper(idx,dataframe):
    inputs=[]
    columnes=dataframe.columns
    for i in range(dataframe.shape[1]):
        if columnes[i]=='Familia' or columnes[i]=='Stopers' or columnes[i]=='Sliders' or columnes[i]=='Teeth' or columnes[i]=='Color' or columnes[i]=='Llargada' or columnes[i]=='Label' :
            keys=str(idx+1)+ str(i+1)
            #input=st.number_input()
            text=st.text_input('Insert the'+ columnes[i]+ 'Code:',key=keys)
            inputs.append(text)
    df=pd.DataFrame([inputs], columns=['Familia', 'Stopers', 'Sliders', 'Label','Teeth', 'Color', 'Llargada'])    
    return df

def print(df_not_encoded):
    st.write(df_not_encoded)


def zippers_model(df, dataset,df_not_encoded, important ,predict, quartils,relative_error_mean_reduced,previa):
    search_zipper=insert_zipper(-1,df_not_encoded)
    finish_zipper=st.button('Submit Zipper')
    if finish_zipper: 
        counter=0
        for i in range(len(df_not_encoded)):
            if (df_not_encoded['Familia'].iloc[i]==search_zipper.iloc[0][0]):
                if (df_not_encoded['Stopers'].iloc[i]==search_zipper.iloc[0][1]): 
                    if (df_not_encoded['Sliders'].iloc[i]==search_zipper.iloc[0][2]):
                        if (df_not_encoded['Label'].iloc[i]==search_zipper.iloc[0][3]):
                            if (df_not_encoded['Teeth'].iloc[i]==search_zipper.iloc[0][4]):
                                if (df_not_encoded['Color'].iloc[i]==search_zipper.iloc[0][5]):
                                    if (df_not_encoded['Llargada'].iloc[i]==search_zipper.iloc[0][6]):
                                        counter=i
                                        st.write('The zipper exists in the data!')
                                        st.write('The zipper introduced is: ', search_zipper)
                
        counter=int(counter)
        aux=df.iloc[counter]
        #aux=pd.DataFrame(aux)
        #aux = np.transpose(aux)
        #column_names = aux.columns.tolist()

    # Create a new list of column names starting from column 7
        #new_column_names = [i if i >= 7 else column_names[i] for i in range(len(column_names))]
    # Assign the new column names to the DataFrame
        #aux.columns = new_column_names   
        #aux1=aux.copy()
        #aux=aux.drop(int(previa),axis=1)
        aux=aux[int(previa)]        
        #new_data=new_data=array_to_dataset(df_result,len(df_result),dataset)
        
            #new_prediction_multiclass= multiclass_model.predict(aux)
            #st.write(new_prediction_multiclass)
        if aux==0: 
            st.write('The zipper will be sold around this quantity: ', 0 ,'-',quartils[0])
        elif aux==1:
            st.write('The zipper will be sold around this quantity: ', quartils[0],'-',quartils[1])
        elif aux==2: 
            st.write('The zipper will be sold around this quantity: ', quartils[1] ,'-',quartils[2])
        else: 
            st.write('The zipper will be sold around this quantity: ', quartils[2] ,'-',quartils[3])
        


@st.cache_data
def truncate(number, max_decimals):
    int_part, dec_part = str(number).split(".")
    return float(".".join((int_part, dec_part[:max_decimals])))

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


def convert_df(df):
   return df.to_csv().encode('utf-8')

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


dataset=st.session_state.data
dataset_grouped=st.session_state.data_encoded


num=st.session_state.numero
df_not_encoded=st.session_state.data_not_encoded
df_regression=st.session_state.df_regression
array_quartils=st.session_state.array_quartils
df=st.session_state.df
quartilss=st.session_state.quartils
values_dict=st.session_state.diccionari
previa1=st.session_state.previa1
df_not_encoded11=st.session_state.df_model


#important=st.session_state.important
important=1
previa=st.session_state.num_mes_previ
previa_per_quartils=int(previa[0])-7

dataset=convert_to_string(dataset_grouped)


#starting plotting
st.title('Zipper prediction with ML')
st.write('')
st.write("We are going to see some predictions of our datasets using Machine Learning and Shap plots")
st.write('You will be able to see do two different types of prediction:')
st.write('1-You will be able to see the prediction of one zipper for the next month.')
st.write('2-Prediction for all the data. ')
st.write('')
st.write('')
st.write('Here you can find the prediction and the explainability of the model. ')


def get_quartils(previa_per_quartils,array_quartils):
    quartils=pd.DataFrame(array_quartils[previa_per_quartils])
    return quartils


model_regression,X_test_regression_reduced,x_regression_reduced,Y_test_regression_reduced, relative_error_mean_reduced,predict=apply_regression(df_regression,previa1)
#predict_xgb,model_xgboost,X_train2, X_test2, Y_train2, Y_test2,x2,y2=multiclass_classification(df,previa1)


with st.expander("Prediction"):
    tab1,tab2= st.tabs(["Prediction of a zipper already in the dataset", " Prediction Dataset for the next months"])
    with tab1: 
        st.write('You will do a prediction with a zipper already in the dataset!')
        
        zippers_model(df,dataset_grouped,df_not_encoded, important,predict, quartilss, relative_error_mean_reduced,previa1)
    with tab2:  
        st.write('Prediction for the next months: ')
        st.write('')
            #df_encoded,df_not_encoded=model(df_regression, LinearRegression(),values_dict,num)
        st.write('The prediction of the data is: ')
        df_not_encoded1=df_not_encoded.copy()
        st.write(df_not_encoded11)
        csv = convert_df(df_not_encoded1)
        st.download_button(label="Download data as CSV",data=csv,file_name='Regression_predictions.csv',mime='text/csv')
