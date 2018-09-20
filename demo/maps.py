import plotly
plotly.tools.set_credentials_file(username='sithuc', api_key='jQAWvmf3T2kGGMJRsFgg')
import dash
import json
import dash_core_components as dcc
import dash_html_components as html
import plotly.plotly as py
import plotly.graph_objs as go
import base64

mapbox_access_token = 'pk.eyJ1Ijoic2l0aHVjIiwiYSI6ImNqbDR6dG50MTJocG0zcXFrdXhuYXZiNTUifQ.Pd7MeCLIK97AcnGgTBvSbA'
dir = "../output"
file= "/output_43729_50222.json"

list_color = ['rgb(17,157,255)','rgb(255, 153, 255)','rgb(0, 0, 102)','rgb(255, 0, 102)',
              'rgb(51, 204, 204)','rgb(255, 255, 153)','rgb(255, 153, 204)','rgb(0, 153, 0)',
              'rgb(255, 153, 0)','rgb(0, 204, 153)','rgb(102, 102, 51)','rgb(153, 51, 0)']

with open(dir + file, 'r') as f:
    arrayObject = json.load(f)
data = []
color = 0

num = 1
number = []
tweets = []
score = []
for cluster in arrayObject:
    number.append(num)
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

    tweets.append(list_members)
    data.append(
        go.Scattermapbox(
            lon=lng,
            lat=lat,
            mode='markers',
            marker=dict(
                size = 9,
                color = list_color[color],
            ),
            text=text,
            name = "Event-" + str(num),
        )
    )

    num += 1
    color += 1
    if color >11:
        color=11



layout = go.Layout(
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat= 37,
            lon=-93.5
        ),
        pitch=1,
        zoom = 2,
    ),
)

table_trace = go.Table(
        columnwidth = [7,7,70],
        header = dict(values = ["Event No.","Score","Tweets"],
                      font=dict(size=12),
                      fill=dict(color='#119DFF')),
        cells = dict(values = [number,score,tweets])
    )
data_table = [table_trace]

#image
image_filename = "../image/hannover.PNG" # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app = dash.Dash()
app.layout = html.Div([
    html.Div(children = [
        html.H1(children = "Disaster Events Detector"),
        html.P(children = 'A Web interface for displaying disaster events in the U.S.A')
    ],style={
        'background-color':'lightblue'
    }),
    html.Div([
        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
    ]),
    html.Div(children=[
        html.Div(children=[
            dcc.Graph(
                id = 'map',
                figure = dict(data=data, layout=layout)
            )
        ],style={
            'width':'45%',
            'height':'auto',
            'float':'left'
        }),
        html.Div(children=[
            dcc.Graph(
                id = "table",
                figure= go.Figure(data=data_table)
            )
        ],style={
        'margin':'auto',
        'width':'55%',
        'height':'auto',
        'float':'left'
        })
    ],style={
        'background-color': '#d854cd'
    }),
    html.Div(children = 'Designed by Si Thuc Pham - ITIS',
             style = {
                'height':'auto',
                 'background-color': 'rgb(78, 244, 66)'

             })
],
style = {
    'text-align':'center',
    'width':'1300px',
    'height':'auto',
    'margin':'auto'
})

if __name__ == '__main__':
    app.run_server(debug=True)
