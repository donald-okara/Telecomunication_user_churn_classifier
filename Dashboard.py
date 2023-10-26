# Import necessary libraries
import dash
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
import plotly.figure_factory as ff  # Import the violin module



# Initialize the Dash app
app = dash.Dash(__name__)

# Read your data
df = pd.read_csv('telecom_customer_churn.csv')  # Replace 'your_file.csv' with your CSV file path

df['Customer Status'] = np.where((df['Customer Status'] != 'Churned'), 'Not Churned', df['Customer Status'])

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


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Define the features and label
features = cat_cols + num_cols
label = 'Customer Status'

# Convert the label column to ordinal categories
label_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

y = label_encoder.fit_transform(df[label].values.reshape(-1, 1))

# Reshape the y variable using ravel
y = y.ravel()


# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[features], y, test_size=0.3, random_state=0)

# Define a pipeline for numeric columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Define a pipeline for categorical columns
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# Create a ColumnTransformer to apply the pipeline to the numeric and categorical columns
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# Fit the preprocessor to the training data and transform both the training and test data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)


# Create a dictionary to map encoded variables to original labels
label_mapping = {}
for i, label in enumerate(label_encoder.categories_[0]):
    label_mapping[i] = label



#Import classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

rf_model = RandomForestClassifier(random_state=1).fit(X_train_transformed, y_train)  # Convert one-hot encoded y_train to 1D array


#Model evaluation
#from preprolib.myfunctions import evaluate_model

#evaluate_model(rf_model,X_test_transformed, y_test)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    

predictions = rf_model.predict(X_test_transformed)

# Calculate feature importance using a trained RandomForestClassifier
feature_importance = rf_model.feature_importances_

# Combine feature names and their importance scores
feature_importance_data = pd.DataFrame({'Feature': features, 'Importance': feature_importance})

# Sort the features by importance (descending order)
feature_importance_data = feature_importance_data.sort_values(by='Importance', ascending=False)




#evaluate_model(rf_model,X_test_transformed, y_test)



# Reshape the test_pred array to a 2D array
predictions = predictions.reshape(-1, 1)

predictions = label_encoder.inverse_transform(predictions)


# Reshape the test_pred array to a 2D array
y_test = y_test.reshape(-1, 1)

y_test = label_encoder.inverse_transform(y_test)
y_test

# Create a placeholder for the confusion matrix plot
confusion_matrix_plot = dcc.Graph(id='confusion-matrix-plot')

# Define the layout of the app
app.layout = html.Div([
    html.H1(id='H1', children='Telecommunication User Churn Dashboard', style={'textAlign': 'left',
                                                                               'marginTop': 40, 'marginBottom': 40}),
                                                                               
    html.H2(id='H2', children='Distribution of Variables.', style={'textAlign': 'left',
                                                                               'marginTop': 40, 'marginBottom': 40}),
    
    # 1. Distribution of the Target Variable (Customer Status)
    dcc.Graph(id='status-distribution'),

    # 2. Distribution of Other Columns (Histograms)
    html.Div(id='histogram-subplots'),
    
    # 3. Comparison Plots of Features Versus the Target Variable
    html.H2(id='H3', children='Comparison Pie Charts of Categorical Features', style={'textAlign': 'left',
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

    html.H2(id='H4', children='Comparison Box Plots of Numerical Features', style={'textAlign': 'left',
                                                                      'marginTop': 20, 'marginBottom': 20}),
    
    # Numeric feature comparison (Box plots)
    html.Div(id='box-plots-container', style={'display': 'flex', 'flex-wrap': 'wrap'}),
    
    html.H2(id='H5', children='This is the confusion matrix for the model.', style={'textAlign': 'left',
                                                                      'marginTop': 20, 'marginBottom': 20}),

    confusion_matrix_plot,
# Inside the app.layout definition
    dcc.Graph(id='feature-importance-plot'),

    html.H2(id='H6', children='These are the Cross Validation scores of the model.', style={'textAlign': 'left',
                                                                      'marginTop': 20, 'marginBottom': 20}),
    dcc.Markdown(
        id='P2',
        children='''Cross-validation Scores: 0.82344428  
                     0.84057971  
                     0.81884058  
                     0.81014493  
                     Average Cross-validation Accuracy: 0.8264279871641603''',
        style={'textAlign': 'left', 'marginTop': 20, 'marginBottom': 20}
    ),

    html.H2(id='H7', children='These are the Evaluation scores of the model.', style={'textAlign': 'left',
                                                                      'marginTop': 20, 'marginBottom': 20}),
    dcc.Markdown(
        id='P3',
        children='''Model Evaluation: Accuracy: 0.8248816768086545  
                     Precision: 0.8192510046389897  
                     Recall: 0.8248816768086545  
                     F1-score: 0.8207080313395132''',
        style={'textAlign': 'left', 'marginTop': 20, 'marginBottom': 20}
    ),
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


# Callback to update the confusion matrix plot
@app.callback(
    Output('confusion-matrix-plot', 'figure'),
    Input('status1-dropdown', 'value'),
    Input('status2-dropdown', 'value')
)
def update_confusion_matrix_plot(status1, status2):
    # Calculate the confusion matrix as you did before
    confusion = confusion_matrix(y_test, predictions)
    confusion_df = pd.DataFrame(confusion, index=["Actual " + str(label) for label in np.unique(y_test)],
                                columns=["Predicted " + str(label) for label in np.unique(y_test)])
    
    # Define the custom blue color scale
    color_scale = [[0, '#FFFFFF'], [0.1, '#D9F0FF'], [0.5, '#77B2E5'], [1, '#003399']]
    
    # Create the confusion matrix heatmap
    fig = px.imshow(confusion_df, title="Confusion Matrix", color_continuous_scale=color_scale)
    fig.update_xaxes(title_text="Predicted Labels")
    fig.update_yaxes(title_text="Actual Labels")
    
    return fig


# Add this callback function at the end of your code

@app.callback(
    Output('feature-importance-plot', 'figure'),
    Input('status1-dropdown', 'value'),
    Input('status2-dropdown', 'value')
)
def update_feature_importance_plot(status1, status2):
    # Calculate feature importance using a trained RandomForestClassifier
    feature_importance = rf_model.feature_importances_

    # Combine feature names and their importance scores
    feature_importance_data = pd.DataFrame({'Feature': features, 'Importance': feature_importance})

    # Sort the features by importance (descending order)
    feature_importance_data = feature_importance_data.sort_values(by='Importance', ascending=False)

    # Create a bar chart to visualize feature importance
    fig = px.bar(
        feature_importance_data,
        x='Feature',
        y='Importance',
        title='Feature Importance',
    )

    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

