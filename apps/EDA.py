# Import Libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pathlib
import plotly
import plotly.express as px
import plotly.graph_objects as go

import flask
import glob
import os

import numpy as np
import pandas as pd
import seaborn as sns
import nltk 
sns.set(style = "darkgrid",font_scale = 1.2)
import matplotlib.pyplot as plt
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from app import app
import dash_bootstrap_components as dbc




#-----------------------------------------------------------------------------
# Import Data
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

template = "seaborn"


#Owner :
jobs = pd.read_csv(DATA_PATH.joinpath("fake_job_postings.csv"))
jobs = jobs.replace(np.nan, '', regex=True)


jobscopy = pd.read_csv(DATA_PATH.joinpath("fake_job_postings.csv"))
jobscopy = jobscopy.replace(np.nan, '', regex=True)



jobs['desc_chars'] = jobs['description'].str.len()
jobs['comp_chars'] = jobs['company_profile'].str.len()
jobs['desc_chars'] = pd.to_numeric(jobs['desc_chars'])
jobs['comp_chars'] = pd.to_numeric(jobs['comp_chars'])

# Joining all relevant columns together
jobs_alltext = jobs

jobs_alltext["all_text"] = jobs["title"] + " " + jobs["location"] + " " + jobs["department"] + " " + jobs["company_profile"] + " " + jobs["description"] + " " + jobs["requirements"] + " " + jobs["benefits"] + " " + jobs["employment_type"] + " " + jobs["required_experience"] + " " + jobs["required_education"] + " " + jobs["industry"] + " " + jobs["function"]
        
#Removing unused columns
del jobs_alltext["title"] 
del jobs_alltext["location"] 
del jobs_alltext["department"]
del jobs_alltext["company_profile"] 
del jobs_alltext["description"] 
del jobs_alltext["requirements"] 
del jobs_alltext["benefits"] 
del jobs_alltext["employment_type"] 
del jobs_alltext["required_experience"] 
del jobs_alltext["required_education"] 
del jobs_alltext["industry"]
del jobs_alltext["function"]
del jobs_alltext["job_id"]
del jobs_alltext["salary_range"]

#Remove Stop Words
from nltk.corpus import stopwords
stop = stopwords.words('english')

jobs_alltext['all_text'] = jobs_alltext.all_text.str.replace("[^\w\s]", "").str.lower()
jobs_alltext['all_text'] = jobs_alltext['all_text'].apply(lambda x: [item for item in x.split() if item not in stop])
jobs_alltext['all_text']= jobs_alltext['all_text'].str.join(" ") 


#Ngrams


og = pd.read_csv(DATA_PATH.joinpath("og.csv"))
bg = pd.read_csv(DATA_PATH.joinpath("bg.csv"))
tg = pd.read_csv(DATA_PATH.joinpath("tg.csv"))

onegram = px.bar(og, y="Word", x="Count",orientation="h",template = template,color="Count", color_continuous_scale="rainbow")
bigram = px.bar(bg,y="Word", x="Count",orientation="h",template = template,color="Count", color_continuous_scale="rainbow")
trigram = px.bar(tg,y="Word", x="Count",orientation="h",template = template,color="Count", color_continuous_scale="rainbow")

#-----------------------------------------------------------------------------

layout = html.Div([
    
    
       dbc.Row( html.Div(html.H1('EDA'))),
       

   dbc.Row([ 
       
       dbc.Col(html.H2('Bar Plots')), dbc.Col(html.H2('Violin Plots'))]),
      

   
   dbc.Row([
   
       
   dbc.Col(
       dcc.Dropdown(
        id='figdropdown',
        options=[
                {'label': 'Fraudulent', 'value': 'fraudulent'},
                {'label': 'Employment Type', 'value': 'employment_type'},
                {'label': 'Education Requirements', 'value': 'required_education'},
        ],
        value='employment_type', clearable=False,)),
       
       dbc.Col(
       dcc.Dropdown(
            id='violindrop',
            options=[
                {'label': 'Description Characters', 'value': 'desc_chars'},
                {'label': 'Company Profile Characters', 'value': 'comp_chars'},
            ],
            value='desc_chars'
        ))]),
       
   
   dbc.Row([
   dbc.Col(
           dcc.Graph(id='fig')),
       
       
       dbc.Col(
       dcc.Graph(id='violinplot'))
        ])   ,             



       # Div 4 - Ngrams
       dbc.Row(html.Div(html.H2('Nrams'))),
         
       dbc.Row([
        dbc.Col(dcc.Graph(figure= onegram)),
          dbc.Col(dcc.Graph(figure= bigram)),
           dbc.Col(dcc.Graph(figure= trigram)),
                 
       ]
       )
           ])   
              



#-----------------------------------------------------------------------------

@app.callback(
    dash.dependencies.Output('fig', 'figure'),
    [dash.dependencies.Input('figdropdown', 'value')])
def update_fig(value):
    fig1 = px.histogram(jobscopy, x=value,color='fraudulent',template = template,color_discrete_sequence=px.colors.qualitative.Set1)
    return fig1



@app.callback(
    Output('violinplot', 'figure'), [
    Input('violindrop', 'value')])
def update_violin(value):
    return {
        'data': [
            {
                'type': 'violin',
                'y': jobs[value],
                'x': jobs['fraudulent'],
                'template': template,
            }
        ],
        'layout': {
            'margin': {'l': 30, 'r': 10, 'b': 30, 't': 0}
        }
    }

