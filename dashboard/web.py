from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

df = pd.read_csv('./movie_char_big_five_predict_result_organized.csv')

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='Title of Dash App', style={'textAlign':'center'}),
    dcc.Dropdown(df.movie_name.unique(), '', id='dropdown-selection'),
])
if __name__ == '__main__':
    app.run(debug=True)
