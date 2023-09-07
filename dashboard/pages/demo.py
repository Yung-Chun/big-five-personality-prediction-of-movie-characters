import dash
from dash import html, dcc, callback, Input, Output, State
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc

dash.register_page(__name__)

data = pd.read_csv('./movie_char_big_five_predict_result_organized.csv')
data = data.reindex(columns=[
    'movie_name', 'char', 'Openness', 'Conscientiousness', 'Extraversion',
    'Agreeableness', 'Neuroticism'
])
personality_prob = [
    'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness',
    'Neuroticism'
]

layout = html.Div(id="demo-area",
                  children=[
                      dbc.Row([
                          dbc.Col([
                              html.H3('Select a movie',
                                      style={
                                          'font-family': 'Times New Roman',
                                          'marginTop': 4
                                      }),
                              dcc.Dropdown(
                                  id='dropdown-selection-movie',
                                  placeholder='Select a movie',
                                  options=data.movie_name.unique(),
                              ),
                          ]),
                          dbc.Col([
                              html.H3('Select a character',
                                      style={
                                          'font-family': 'Times New Roman',
                                          'marginTop': 4
                                      }),
                              dcc.Dropdown(
                                  id='dropdown-selection-character',
                                  placeholder='Select a character',
                                  options=[],
                              ),
                          ])
                      ]),
                      html.H3(' '),
                      dbc.Row([
                          dbc.Col(dcc.Graph(id='big-five-radar', figure={})),
                          dbc.Col(dbc.Row(id='big-five-info', children=[])),
                      ]),
                  ])


@callback(
    Output('dropdown-selection-character', 'options'),
    Input('dropdown-selection-movie', 'value'),
)
def set_character_options(movieName):
    options = data[data['movie_name'] == movieName].char.unique()
    return options


@callback(
    Output('big-five-radar', 'figure'),
    Input('dropdown-selection-movie', 'value'),
    Input('dropdown-selection-character', 'value'),
)
def draw_radar(movieName, characterName):
    if characterName == None:
        prob = []
    else:
        if characterName in data.query(
                f'movie_name == "{movieName}"')['char'].to_list():
            prob = data.query(f'movie_name == "{movieName}"').query(
                f'char == "{characterName}"'
            )[personality_prob].iloc[0].to_list()
        else:
            prob = []

    fig = px.line_polar(
        r=prob,
        theta=personality_prob,
        line_close=True,
        color_discrete_sequence=['lightcoral'],
        template='plotly_dark',
        height=400,
        width=600,
    )
    fig.update_traces(fill='toself')
    fig.update_polars(
        angularaxis_showgrid=True,
        radialaxis_gridwidth=0,
        gridshape='linear',
        bgcolor="#494b5a",
        radialaxis_showticklabels=True,
    )
    # fig.update_polars(angularaxis_dtick='')
    fig.update_layout(
        font=dict(
            size=14,  # Set the font size here
            family='Times New Roman',
        ),
        # paper_bgcolor="#2c2f36",
        paper_bgcolor='black',
        polar=dict(radialaxis=dict(visible=True, range=[0, 1]), ),
        showlegend=False)
    return fig


@callback(
    Output('big-five-info', 'children'),
    Input('dropdown-selection-movie', 'value'),
    Input('dropdown-selection-character', 'value'),
)
def write_info(movieName, characterName):
    if characterName == None:
        prob = []
    else:
        if characterName in data.query(
                f'movie_name == "{movieName}"')['char'].to_list():
            prob = data.query(f'movie_name == "{movieName}"').query(
                f'char == "{characterName}"'
            )[personality_prob].iloc[0].to_list()
            text_style = {
                'color': 'white',
                'font-size': 16,
                'marginTop': 14,
                'font-family': 'Times New Roman'
            }
        else:
            return []

        return [
            html.Div([
                html.Div([
                    html.Div('The Big-Five personality prediction of ' +
                             characterName + ' in ' + movieName + ' is:')
                ],
                         style={
                             'color': 'white',
                             'marginTop': 14,
                             'font-size': 20
                         }),
                html.Div([
                    html.Div(
                        personality_prob[0] + ": " + str(round(prob[0], 4)),
                        style=text_style),
                    html.Div(
                        personality_prob[1] + ": " + str(round(prob[1], 4)),
                        style=text_style),
                    html.Div(
                        personality_prob[2] + ": " + str(round(prob[2], 4)),
                        style=text_style),
                    html.Div(
                        personality_prob[3] + ": " + str(round(prob[3], 4)),
                        style=text_style),
                    html.Div(
                        personality_prob[4] + ": " + str(round(prob[4], 4)),
                        style=text_style),
                ]),
            ], ),
        ]
