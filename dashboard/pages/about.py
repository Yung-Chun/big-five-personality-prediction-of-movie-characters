import dash
from dash import html, dcc
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

dash.register_page(__name__, path='/')

data = pd.read_csv('./movie_char_big_five_predict_result_organized.csv')

all_records = pd.DataFrame()
y_cols = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']

root_path = Path(fr'/Users/ycchen/Desktop/UCSD_CSS/summer_project/big_five')
for file in root_path.glob(f'*svc_essays_embedding*linear*50*'):

    temp = pd.read_csv(root_path / f'{file.stem}.csv')
    temp['file'] = file.stem.split('svc_essays_embedding_')[1]
    all_records = pd.concat([all_records, temp])

all_records = all_records.replace('cCON', 'Conscientiousness')\
                         .replace('cEXT', 'Extraversion')\
                         .replace('cNEU', 'Neuroticism')\
                         .replace('cAGR', 'Agreeableness')\
                         .replace('cOPN', 'Openness')

new_y_cols = [
    'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness',
    'Neuroticism'
]

colors = [
    'navajowhite', 'lightsteelblue', 'darkseagreen', 'darksalmon', 'indianred'
]
fig = go.Figure()

for n in range(0, 5):

    fig.add_trace(
        go.Scatter(
            x=all_records[all_records.y == new_y_cols[n]]['C'],
            y=all_records[all_records.y == new_y_cols[n]]['mean_train_score'],
            name=f'Mean Train Score of {new_y_cols[n]}',
            mode='lines+markers',
            marker=dict(symbol='circle', size=6),
            line=dict(width=2, color=colors[n])))

    fig.add_trace(
        go.Scatter(
            x=all_records[all_records.y == new_y_cols[n]]['C'],
            y=all_records[all_records.y == new_y_cols[n]]['mean_test_score'],
            name=f'Mean Test Score of {new_y_cols[n]}',
            mode='lines+markers',
            marker=dict(symbol='x', size=6),
            line=dict(width=2, color=colors[n])))

fig.update_layout(
    height=600,
    width=1000,
    title='Performance of Each Personality Trait Prediction',
    title_font_family='Times New Roman',
    xaxis_title='C',  # y-axis name
    yaxis_title='Accuracy',  # x-axis name
    template='plotly_dark',
    paper_bgcolor='black',
    legend=dict(font=dict(family='Times New Roman')))

fig.update_xaxes(title_font=dict(family='Times New Roman'))
fig.update_yaxes(title_font=dict(family='Times New Roman'))

text_style1 = {
    'color': 'white',
    'marginTop': 30,
    'font-size': 18,
    'font-family': 'Garamond'
}
text_style2 = {
    'color': 'white',
    'marginTop': 10,
    'marginLeft': 60,
    'font-size': 18,
    'font-family': 'Times New Roman'
}
layout = html.Div([
    html.H3('This is the final project of CSS Bootcamp',
            style={
                'marginTop': 18,
                'font-family': 'Times New Roman',
            }),
    html.Div(
        [
            """
        The Big Five personality model developed in psychological trait theory since 1980s.
        In 1990s, the theory indicated 5 factors as below:
        """
        ],
        style=text_style1,
    ),
    html.Div(
        ['Openness to experience (inventive/curious vs. consistent/cautious)'],
        style=text_style2),
    html.Div(
        ['Conscientiousness (efficient/organized vs. extravagant/careless)'],
        style=text_style2),
    html.Div(['Extraversion (outgoing/energetic vs. solitary/reserved)'],
             style=text_style2),
    html.Div(['Agreeableness (friendly/compassionate vs. critical/rational)'],
             style=text_style2),
    html.Div([' Neuroticism (sensitive/nervous vs. resilient/confident)'],
             style=text_style2),
    html.Div([
        """
    The training data in this project is provided by Pennebaker & King (1999).
    This a large dataset of 2400 stream-of-consciousness texts labelled with Big-Five personality.
    See: 
    """,
        html.
        A('source',
          href=
          'https://web.archive.org/web/20160519045708/http://mypersonality.org/wiki/doku.php?id=wcpr13'
          )
    ],
             style=text_style1),
    html.Div([
        """
    The data used for prediction id from 
    """,
        html.A("Internet Movie Database ", href='https://www.imdb.com'), f"""
    . In this project, I only compute characters whose number of lines exceeds the 95th percentile.
    There are {len(set(data.movie_name))} movies and {len(set(data.char))} characters in total.
    """
    ],
             style=text_style1),
    html.Div([
        'The prediction is based on support vector classification. The performance of the model is as below:'
    ],
             style=text_style1),
    html.Div(dcc.Graph(id='performance', figure=fig))
])
