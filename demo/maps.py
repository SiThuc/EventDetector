import plotly
from docutils.io import Output

plotly.tools.set_credentials_file(username='sithuc', api_key='jQAWvmf3T2kGGMJRsFgg')
import dash
import json
import dash_core_components as dcc
import dash_html_components as html
import plotly.plotly as py
import plotly.graph_objs as go
import base64
import glob
import os
import time
from dash.dependencies import Input, Output

list_color = ['rgb(17,157,255)','rgb(255, 153, 255)','rgb(0, 0, 102)','rgb(255, 0, 102)',
              'rgb(51, 204, 204)','rgb(255, 255, 153)','rgb(255, 153, 204)','rgb(0, 153, 0)',
              'rgb(255, 153, 0)','rgb(0, 204, 153)','rgb(102, 102, 51)','rgb(153, 51, 0)']

mapbox_access_token = 'pk.eyJ1Ijoic2l0aHVjIiwiYSI6ImNqbDR6dG50MTJocG0zcXFrdXhuYXZiNTUifQ.Pd7MeCLIK97AcnGgTBvSbA'
list_of_files = glob.glob('../output/live/*.json')

app = dash.Dash()
app.layout = html.Div([
    html.Div(children=[
        html.H1(children="Disaster Events Detector"),
        html.P(children='A Web interface for displaying disaster events in the U.S.A')
    ],style={
        'background-color': 'lightblue'
    }),
    html.Div([
        html.Div(children=[
            dcc.Graph(id='live-update-graph')
        ], style={
            'width': '45%',
            'height': 'auto',
            'float': 'left'
        }),
        html.Div(children=[
            dcc.Graph(id="live-update-text")
        ], style={
            'margin': 'auto',
            'width': '55%',
            'height': 'auto',
            'float': 'left'
        })
    ],style={
        'background-color': '#d854cd'
    }),
    dcc.Interval(
        id='interval-component',
        interval=3 * 1000,  # in milliseconds
        n_intervals=0
    ),
    html.Div(children='Designed by Si Thuc Pham - ITIS',
    style={
         'height': 'auto',
         'background-color': 'rgb(78, 244, 66)'

    })
],style = {
    'text-align':'center',
    'width':'1300px',
    'height':'auto',
    'margin':'auto'
})

@app.callback(Output('live-update-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_maps(n):
    file = max(list_of_files, key=os.path.getctime)
    with open(file, 'r') as f:
        arrayObject = json.load(f)
    f.close()
    data = []
    color = 0
    num = 1
    score = []
    for cluster in arrayObject:
        score.append(cluster['Score'])
        list_members = ''
        lat = []
        lng = []
        text = []
        for mem in cluster['Members']:
            lng.append(str(mem['lng']))
            lat.append(str(mem['lat']))
            list_text = mem['text']
            temp = " ".join(list_text)
            text.append(temp)
            list_members += str(mem['num']) + ". " + temp + "\n\n"

        data.append(
            go.Scattermapbox(
                lon=lng,
                lat=lat,
                mode='markers',
                marker=dict(
                    size=9,
                    color=list_color[color],
                ),
                text=text,
                name="Event-" + str(num),
            )
        )

        num += 1
        color += 1
        if color > 11:
            color = 11

    layout = go.Layout(
        hovermode='closest',
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=dict(
                lat=37,
                lon=-93.5
            ),
            pitch=1,
            zoom=2,
        ),
    )
    fig = dict(data= data, layout = layout)
    f.close()
    return fig

@app.callback(Output('live-update-text', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_table(n):
    number = []
    tweets = []
    num = 1
    file = max(list_of_files, key=os.path.getctime)
    with open(file, 'r') as f:
        arrayObject = json.load(f)
    f.close()
    for cluster in arrayObject:
        list_members = ''
        lat = []
        lng = []
        text = []
        for mem in cluster['Members']:
            lng.append(str(mem['lng']))
            lat.append(str(mem['lat']))
            list_text = mem['text']
            temp = " ".join(list_text)
            text.append(temp)
            list_members += str(mem['num']) + ". " + temp + "\n\n"
        number.append(num)
        tweets.append(list_members)
        num += 1


    table_trace = go.Table(
        columnwidth=[7, 70],
        header=dict(values=["Event No.", "Tweets"],
                    font=dict(size=12),
                    fill=dict(color='#119DFF')),
        cells=dict(values=[number, tweets])
    )
    data_table = [table_trace]

    fig = go.Figure(data = data_table)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

