# Import necessary libraries
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

# Separate numerical and categorical columns
from preprolib import myfunctions
num_cols = []
cat_cols = []

ignore_list = ['Zip Code', 'Longitude', 'Latitude', 'Customer ID', 'Churn Category', 'Churn Reason', 'Customer Status', 'City']

myfunctions.cat_or_num(df, ignore_list, num_cols, cat_cols)

# Remove specific columns from num_cols and cat_cols
remove_num = ['Age', 'Avg Monthly Long Distance Charges', 'Avg Monthly GB Download', 'Monthly Charge']
num_cols = [x for x in num_cols if x not in remove_num]

remove_cat = ['Gender', 'Married', 'Multiple Lines']
cat_cols = [x for x in cat_cols if x not in remove_cat]

# Define the layout of the app
app.layout = html.Div([
    html.H1(id='H1', children='Telecommunication User Churn Dashboard', style={'textAlign': 'center',
                                                                               'marginTop': 40, 'marginBottom': 40}),
    
    # 1. Distribution of the Target Variable (Customer Status)
    dcc.Graph(id='status-distribution'),
    
    # 2. Distribution of Other Columns (Histograms)
    html.Div(id='histogram-subplots'),
    
    # 3. Comparison Plots of Features Versus the Target Variable
    html.H1(id='H3', children='Comparison Plots of Features', style={'textAlign': 'left',
                                                                      'marginTop': 20, 'marginBottom': 20}),
    
    # Dropdowns for feature selection and status selection
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
    
    # Categorical feature comparison (Pie charts)
    html.Div(id='pie-charts-container', style={'display': 'flex', 'flex-wrap': 'wrap'}),

    # Numeric feature comparison (Box plots)
    html.Div(id='box-plots-container', style={'display': 'flex', 'flex-wrap': 'wrap'}),
    
    # Continue with your existing layout...
])

# 1. Distribution of the Target Variable (Customer Status)
@app.callback(
    Output('status-distribution', 'figure'),
    Input('status1-dropdown', 'value'),
    Input('status2-dropdown', 'value')
)
def update_status_distribution(status1, status2):
    # Calculate the distribution of Customer Status
    status_counts = df['Customer Status'].value_counts()
    
    # Create a bar chart for the distribution
    fig = px.bar(
        x=status_counts.index,
        y=status_counts.values,
        labels={'x': 'Customer Status', 'y': 'Count'},
        title='Distribution of Customer Status'
    )
    
    return fig

# 2. Distribution of Other Columns (Histograms)
@app.callback(
    Output('histogram-subplots', 'children'),
    Input('feature-dropdown', 'value')
)
def update_histograms(selected_feature):
    # Define a list of columns you want to display as histograms
    columns_to_display = [cat_cols, num_cols]  # Modify this list
    
    histogram_subplots = []
    for column in columns_to_display:
        for col in column:
            fig = go.Figure(data=[go.Histogram(x=df[col])])
            fig.update_layout(title=f'Distribution of {col}')
            histogram_subplots.append(dcc.Graph(figure=fig))
    
    return histogram_subplots

# 3. Comparison Plots of Features Versus the Target Variable (Box plots)
@app.callback(
    Output('box-plots-container', 'children'),
    Input('feature-dropdown', 'value')
)
def update_box_plots(selected_feature):
    # Define a list of numeric columns you want to display as box plots
    numeric_columns = num_cols  # Modify this list
    
    box_plot_subplots = []
    for column in numeric_columns:
        fig = go.Figure(data=[go.Box(x=df['Customer Status'], y=df[column])])
        fig.update_layout(title=f'{column} vs. Customer Status (Box Plot)')
        box_plot_subplots.append(dcc.Graph(figure=fig))
    
    return box_plot_subplots


# 4. Categorical feature comparison (Pie charts)
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
                        title=f'{category} Distribution',
                        showlegend=False
                    )
                },
                style={'width': '48%', 'height': '300px', 'margin': '1%'}
            )
        )

    return pie_charts



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

