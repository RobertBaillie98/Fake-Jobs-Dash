# Import Libraries
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pathlib
import plotly
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import flask
import glob
import os

import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style = "darkgrid",font_scale = 1.2)


from app import app
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff



#----------------------------------------------------------------------------


# Script in jupyter notebook
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()


#Owner :
df = pd.read_csv(DATA_PATH.joinpath("scores.csv"))
df = df.replace(np.nan, '', regex=True)

dataset = pd.read_csv(DATA_PATH.joinpath("dataset.csv"))

template = "seaborn"
#-----------------------------------------------------------------------------


layout = html.Div([
        dbc.Row(html.Div(html.H1('Models'))),
        
        
    
    
     
     dbc.Row([dbc.Col(html.Div(html.H2('Accuracy and F1'))), dbc.Col(html.Div(html.H2('Confusion Matrix')))]),
        
        
        dbc.Row([
            dbc.Col(
        dcc.Dropdown(
        id='dropdown',
        options=[
                {'label': 'Validation F1', 'value': 'val_f1'},
                {'label': 'Test F1', 'value': 'test_f1'},
                {'label': 'Test Accuracy', 'value': 'test_acc'}
        ])),
        
        dbc.Col(dcc.Dropdown(
        id='cmdropdown',
        options=[
                {'label': 'Multinomial NB', 'value': 'mnb_test'},
                {'label': 'Support Vector Machine', 'value': 'svm_test'},
                {'label': 'Logistic Regression', 'value': 'logreg_test'}
        ],
                value ='mnb_test',
                clearable=False),)
        ]),
        
        
        dbc.Row([
            dbc.Col(
        dcc.Graph(id='plot')),
        
        dbc.Col(
         dcc.Graph(id='cmplot'))]),
        
        
    html.Div([
        html.H2('Building Models'),
        
        dcc.Dropdown(
        id='moddropdown',
        options=[
                {'label': 'Multinomial NB', 'value': 'mnb'},
                {'label': 'Support Vector Machine', 'value': 'svm'},
                {'label': 'Logistic Regression', 'value': 'logreg'}
        ],
                value ='mnb',
                clearable=False),
        
        
        dcc.Markdown(
        id='text')
       
       ])
])


#-----------------------------------------------------------------------------
@app.callback(
    dash.dependencies.Output('text', 'children'),
    [dash.dependencies.Input('moddropdown', 'value')])
def update_text(value):
    if value == "mnb":
        return '''
    
            from sklearn.naive_bayes import MultinomialNB
            mnb = MultinomialNB()
            mnb.fit(X_train, y_train)
            mnb_predict = mnb.predict(X_train)

            from sklearn.metrics import f1_score
            val_f1_mnb = f1_score(mnb_predict, y_train, average='macro')

            mnb_test = mnb.predict(X_test)
            test_f1_mnb = f1_score(mnb_test, y_test, average='macro')

            from sklearn.metrics import accuracy_score
            test_ac_mnb = accuracy_score(mnb_test,y_test)
''' 
    if value == "svm":
        return '''
    
            from sklearn import svm
            svm = svm.SVC()
            svm.fit(X_train, y_train)

            svm_predict = svm.predict(X_train)
            val_f1_svm = f1_score(svm_predict, y_train, average='macro')

            svm_test = svm.predict(X_test)
            test_f1_svm = f1_score(svm_test, y_test, average='macro')

            test_ac_svm = accuracy_score(svm_test,y_test)
'''
    elif value == "logreg":
        return '''
    
            from sklearn.linear_model import LogisticRegression

            logreg = LogisticRegression().fit(X_train,y_train)
            logreg_predict = logreg.predict(X_train)
            val_f1_logreg = f1_score(logreg_predict, y_train, average='macro')

            logreg_test = logreg.predict(X_test)
            test_f1_logreg = f1_score(logreg_test,y_test, average='macro')

            test_ac_logreg = accuracy_score(logreg_test,y_test)
'''    


@app.callback(
    dash.dependencies.Output('plot', 'figure'),
    [dash.dependencies.Input('dropdown', 'value')])
def update_graph(value):
    figure = px.bar(df, x="Model", y = value, color = "Model",template = template,color_discrete_sequence=px.colors.qualitative.Set1)
    return figure


@app.callback(
    dash.dependencies.Output('cmplot', 'figure'),
    [dash.dependencies.Input('cmdropdown', 'value')])

def update_cm(value):
    cm = confusion_matrix(dataset[value], dataset["y_test"])
    fig = px.imshow(cm,
                labels=dict(x="Predicted", y="True Values", color="Productivity",template = template, color_continuous_scale="rainbow"),
                x=['Real','Fake'],
                y=['Real','Fake'])
    return fig
