import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import dash_bootstrap_components as dbc

import sys
sys.path.append('YOUR DIRECTORY')
  
# Connect to main app.py file
from app import app # Importing app from app.py
from app import server #Importing server from app.py

# Connect to your pages
from apps import EDA, Model_Building  # From apps folder import pages






app.layout =  html.Div([

    html.Div([
    dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("EDA", href="/apps/EDA")),
        dbc.NavItem(dbc.NavLink("Models", href="/apps/Model_Building")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("More pages", header=True),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="Fake Jobs Investigation",
    brand_href="/apps/EDA",
    color="#ff0000",
    dark=True,
),
    dcc.Location(id='url',refresh=False),                                      # Fill Pathname with URL
    html.Div(id='page-content',children=[])
]) 
    ])
   



@app.callback(Output(component_id = 'page-content', component_property = 'children'),    #Take URL ID and Pathname (Empty)
              [Input(component_id = 'url', component_property = 'pathname')])           #If pathname is link, return the layout
                                                                                        #and insert the layout
def display_page(pathname):
    if pathname == '/apps/EDA':
        return EDA.layout
    if pathname == '/apps/Model_Building':
        return Model_Building.layout
    else:
        return EDA.layout
    
    
if __name__ == '__main__':
    app.run_server(debug=False)
    
