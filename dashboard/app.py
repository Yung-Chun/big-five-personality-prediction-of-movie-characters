# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, ALL, callback, page_container, State

app = Dash(__name__,
           suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           use_pages=True)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.H1(
        children="Big Five Personality Prediction of Movie Characters",
        style={
            'font-family': 'Georgia',
            'marginTop': 10,
            'marginBottom': 10
        },
        id='my-head',
    ),
    dbc.Row([
        dbc.Col(dbc.Nav(
            [
                dbc.NavItem(
                    dbc.NavLink("About the Dashboard",
                                href="/",
                                style={'font-family': 'Georgia'},
                                id='about-link')),
                dbc.NavItem(
                    dbc.NavLink("Try It Out",
                                href="/demo",
                                style={'font-family': 'Georgia'},
                                id='demo-link'))
            ],
            vertical="md",
            id='my-nav',
        ),
                sm=2,
                id="nav-area"),
        dbc.Col(page_container, id='my-content', sm=10)
    ]),
    html.Div([
        html.Div(
            'Yung-Chun Chen @ University of California, San Diego. Computational Social Science.',
            style={
                'textAlign': 'center',
                'color': 'white',
                'marginTop': 10,
                'fontSize': 14,
                'font-family': 'Times New Roman'
            },
        ),
        html.Div(
            'August 2023',
            style={
                'textAlign': 'center',
                'color': 'white',
                'fontSize': 14,
                'font-family': 'Times New Roman'
            },
        )
    ],
             id='my-footer')
])


@callback(
    [Output('about-link', 'class_name'),
     Output('demo-link', 'class_name')], Input('url', 'pathname'))
def displayPage(pathname):
    if pathname == '/':
        return ['active', '']
    return ['', 'active']


if __name__ == "__main__":
    app.run(debug=True, )
