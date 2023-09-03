import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd

# Initialize the Dash app
app = dash.Dash(__name__)

# Read your data
df = pd.read_csv('telecom_customer_churn.csv')  # Replace 'your_file.csv' with your CSV file path

#Separate numerical and categorical columns
from preprolib import myfunctions
num_cols = []
cat_cols = []

ignore_list = ['Zip Code', 'Longitude', 'Latitude', 
                'Customer ID', 'Churn Category', 
                'Churn Reason', 'Customer Status', 'City']

myfunctions.cat_or_num(df, ignore_list, num_cols, cat_cols)

label = 'Customer Status'

##num_cols.remove

remove_num = ['Age', 'Avg Monthly Long Distance Charges',  'Avg Monthly GB Download', 'Monthly Charge']

num_cols = [x for x in num_cols if x not in remove_num]


##cat_cols.remove

remove_cat = ['Gender', 'Married',  'Multiple Lines']

cat_cols = [x for x in cat_cols if x not in remove_cat]

# Define the layout of the app
app.layout = html.Div([
    html.H1(id = 'H1', children = 'Telecommunication User Curn Dashboard', style = {'textAlign':'center',\
                                            'marginTop':40,'marginBottom':40}),
    html.H1(id = 'H2', children = 'Customer status compared for categorical columns', style = {'textAlign':'left',\
                                            'marginTop':20,'marginBottom':20}),

    html.Label('Feature:'),
    dcc.Dropdown(
        id='feature-dropdown',
        options=[{'label': col, 'value': col} for col in cat_cols],
        value='Offer'
    ),
    
    html.Label('Status 1:'),
    dcc.Dropdown(
        id='status1-dropdown',
        options=[{'label': status, 'value': status} for status in df['Customer Status'].unique()],
        value=df['Customer Status'].unique()[0]
    ),
    
    html.Label('Status 2:'),
    dcc.Dropdown(
        id='status2-dropdown',
        options=[{'label': status, 'value': status} for status in df['Customer Status'].unique()],
        value=df['Customer Status'].unique()[1]
    ),
    
    html.Div(id='pie-charts-container', style={'display': 'flex', 'flex-wrap': 'wrap'})
])

# Define a callback to update the pie charts
@app.callback(
    Output('pie-charts-container', 'children'),
    Input('feature-dropdown', 'value'),
    Input('status1-dropdown', 'value'),
    Input('status2-dropdown', 'value')
)
def update_pie_charts(feature, status1, status2):
    filtered_df = df[(df['Customer Status'].isin([status1, status2]))]

    # Create a subplot for each category within the selected feature
    unique_categories = filtered_df[feature].unique()
    pie_charts = []

    for category in unique_categories:
        category_counts = filtered_df[filtered_df[feature] == category]['Customer Status'].value_counts()
        labels = category_counts.index
        values = category_counts.values

        # Create a pie chart trace for each category
        pie_trace = go.Pie(labels=labels, values=values, hole=0.3)
        
        pie_charts.append(
            dcc.Graph(
                figure={
                    'data': [pie_trace],
                    'layout': go.Layout(
                        title=category,
                        showlegend=False
                    )
                },
                style={'width': '10%', 'height': '100px', 'margin': '0.5%'}
            )
        )

    return pie_charts

if __name__ == '__main__':
    app.run_server(debug=True)
