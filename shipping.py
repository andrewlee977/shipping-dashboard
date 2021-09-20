"""Builds the app and handles all Plotly elements, functions, and visualizations"""

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import pickle
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """Loads data to read into a DataFrame"""
    return pd.read_csv(filepath, index_col="ID")


def load_model(filepath):
    """Loads pretrained model for prediction"""
    return pickle.load(open(filepath, 'rb'))


def split(df):
    """Splits data into X and y"""
    X = df.drop(columns="Reached_on_time")
    y = df["Reached_on_time"]
    return X, y


def split_data(df):
    """Splits data using train_test_split"""
    X, y = split(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2,
                                            random_state=42)
    return X_test, y_test


def perm_imp(model, X_test, y_test):
    """Creates Permutation Importance DataFrame for visualization"""
    perm_imp = permutation_importance(model, X_test, y_test, n_repeats=10,
                                      n_jobs=-1, random_state=42)
    data = {"imp_mean": perm_imp["importances_mean"],
            "imp_std": perm_imp["importances_std"]}
    df_perm = pd.DataFrame(data, index=X_test.columns).sort_values("imp_mean")
    return df_perm


external_stylesheets = [
    dbc.themes.JOURNAL,  # Bootswatch theme
    'https://use.fontawesome.com/releases/v5.9.0/css/all.css',
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True  # see https://dash.plot.ly/urls
app.title = 'Shipping Dashboard'  # appears in browser title bar
server = app.server

# ----------
# Load DataFrame
df = load_data("./data/dash_ready_data.csv")

# Load Model
model = load_model("./model/gbc.pk1")

X_test, y_test = split_data(df)

df_perm = perm_imp(model, X_test, y_test)


# Permutation Importance Graph
def permutation_graph():
    """Graphs permutation feature importances bar graph"""
    fig = go.Figure(go.Bar(
        x=df_perm["imp_mean"],
        y=df_perm.index,
        orientation='h'))

    fig.update_layout(title={"text": "Permutation Feature Importance",
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},
                      xaxis_title="Model Accuracy Decrease",
                      yaxis_title="Features",
                      width=1000,
                      height=500)

    return fig


# ----------------------------------------------------------------------
# App layout
app.layout = html.Div([
    html.Div([
        html.H1("Shipping Dashboard", style={'text-align': 'center'}),
        html.Br(),
        html.Div("""ABOUT THIS DASHBOARD""",
                 style={'color': 'black', 'fontSize': 30}),
        dcc.Link(href="https://www.kaggle.com/prachi13/customer-analytics"),
        dcc.Markdown("""This dashboard was created using E-Commerce
                            shipping data from Kaggle that contains data from
                            an electronic products company. Below are some
                            important insights from exploring the data as well
                            as a predictor at the bottom of the webpage. The 
                            target variable is `Reached_on_time`, which is a
                            binary value representing if a product arrived to
                            the customer on time (1 – Reached on time, 0 – Not
                            Reached on time)."""),
        html.Div("Permutation Feature Importance",
                 style={'color': 'black', 'fontSize': 20}),
        dcc.Markdown("""The permutation feature importance ranks each
                             feature by its influence to the model score. It
                             does so by randomly shuffling every observation
                             for each individual feature and evaluates the
                             drop in model score. The magnitude of the drop
                             in model score is indicative of how much the
                             model depends on that feature in predicting
                             the target variable."""),
        dcc.Markdown("""In our dataset, `Weight_in_gms` accounts for
                             ~6.5% variability in our model score. This means
                             that if this feature were to be randomly
                             shuffled, the model score would drop by 6.5%, on
                             average. This graph is extremely important
                             because it tells us that `Weight_in_gms` is the
                             most important feature and the features
                             contributing less than 1% variablility are
                             negligible and contribute noise to the model."""),
        html.Div("Histogram",
                 style={'color': 'black', 'fontSize': 20}),
        dcc.Markdown("""The histogram illustrates the number of shipments
                             of a chosen continuous feature, split by the 
                             continuous feature's distinct values."""),
        html.Div("Scatter Plot",
                 style={'color': 'black', 'fontSize': 20}),
        dcc.Markdown("""The scatter plot illustrates the relationship
                             between the chosen feature and the target variable
                             `Reached_on_time`. Note that beyond `Weight_in_gms`
                             and `Discount_offered`, a significant relationship
                             is not found."""),
        html.Div("Pie Chart",
                 style={'color': 'black', 'fontSize': 20}),
        dcc.Markdown("""The Pie Chart illustrates the proportions of total
                            shipments by the distinct classes of the chosen
                            categorical feature."""),
        html.Div("Predict Late Shipping",
                 style={'color': 'black', 'fontSize': 20}),
        dcc.Markdown("""Use the interactive dashboard to predict
                             if a product will arrive late. Output: (0 - Predicted
                             Late, 1 = Predicted On Time) and the probability
                             of the shipment being late."""),

        dbc.Row([
            dbc.Col([
                html.Div(
                    "Permutation Feature Importance",
                    style={
                        'color': 'black',
                        'fontSize': 30}),
                dcc.Graph(id='perm_imp', figure=permutation_graph()),
            ]),


            dbc.Col([
                html.Div(
                    "Bar Chart",
                    style={
                        'color': 'black',
                        'fontSize': 30}),
                html.Br(),
                dcc.Dropdown(id="bar_feature",
                             options=[
                                 {"label": "Weight_in_gms",
                                     "value": "Weight_in_gms"},
                                 {"label": "Discount_offered",
                                     "value": "Discount_offered"},
                                 {"label": "Prior_purchases",
                                     "value": "Prior_purchases"},
                                 {"label": "Customer_care_calls",
                                  "value": "Customer_care_calls"},
                                 {"label": "Cost_of_the_Product",
                                  "value": "Cost_of_the_Product"}],
                             placeholder="Choose Feature",
                             style={'width': "60%"},
                             value='Customer_care_calls'
                             ),
                dcc.Graph(id='histogram_graph', figure={}),
            ]),
        ]),


        dbc.Row([
            dbc.Col([
                html.Div(
                    "Scatter Plot",
                    style={
                        'color': 'black',
                        'fontSize': 30}),
                html.Br(),
                dcc.Dropdown(id="feature",
                             options=[
                                 {"label": "Weight_in_gms",
                                     "value": "Weight_in_gms"},
                                 {"label": "Discount_offered",
                                     "value": "Discount_offered"},
                                 {"label": "Prior_purchases",
                                     "value": "Prior_purchases"},
                                 {"label": "Customer_care_calls",
                                  "value": "Customer_care_calls"},
                                 {"label": "Cost_of_the_Product",
                                  "value": "Cost_of_the_Product"}],
                             placeholder="Choose Feature",
                             style={'width': "60%"},
                             value='Weight_in_gms'
                             ),
                dcc.Graph(id='scatter', figure={}),
            ]),
            dbc.Col([
                html.Div(
                    "Pie Chart",
                    style={
                        'color': 'black',
                        'fontSize': 30}),
                html.Br(),
                dcc.Dropdown(id="pie_feature",
                             options=[
                                 {"label": "Warehouse_block",
                                     "value": "Warehouse_block"},
                                 {"label": "Mode_of_Shipment",
                                     "value": "Mode_of_Shipment"},
                                 {"label": "Product_importance",
                                  "value": "Product_importance"},
                                 {"label": "Gender", "value": "Gender"},
                                 {"label": "Reached_on_time", "value": "Reached_on_time"}],
                             placeholder="Choose Feature",
                             style={'width': "60%"},
                             value='Warehouse_block'
                             ),
                dcc.Graph(id='pie_chart', figure={}),
            ]),
        ]),
    ]),
    dbc.Row([
            dbc.Col([
                html.Div("""Predict Late Shipping""",
                         style={'color': 'black', 'fontSize': 30}),
                html.Br(),
                dcc.Dropdown(
                    id='Warehouse_block',
                    options=[
                        {'label': 'A', 'value': 'A'},
                        {'label': 'B', 'value': 'B'},
                        {'label': 'C', 'value': 'C'},
                        {'label': 'D', 'value': 'D'},
                        {'label': 'F', 'value': 'F'}
                    ],
                    placeholder="Warehouse Block"
                ),
                html.Br(),
                dcc.Dropdown(
                    id='Mode_of_Shipment',
                    options=[
                        {'label': 'Ship', 'value': 'Ship'},
                        {'label': 'Flight', 'value': 'Flight'},
                        {'label': 'Road', 'value': 'Road'}
                    ],
                    placeholder="Mode of Shipment"
                ),
                html.Br(),
                dcc.Input(
                    id="Customer_care_calls",
                    placeholder="# Customer Care Calls",
                    type="number",
                    style={'width': '100%'}
                ),
                html.Br(),
                html.Br(),
                dcc.Input(
                    id="Cost_of_the_Product",
                    placeholder="Product Cost",
                    type="number",
                    style={'width': '100%'}
                ),
                html.Br(),
                html.Br(),
                dcc.Input(
                    id="Weight_in_gms",
                    placeholder="Product Weight (Grams)",
                    type="number",
                    style={'width': '100%'}
                ),

            ], md=4),
            dbc.Col([
                html.Br(),
                html.Br(),
                html.Br(),
                dcc.Input(
                    id="Prior_purchases",
                    placeholder="# Prior Purchases",
                    type="number",
                    style={'width': '100%'}
                ),
                html.Br(),
                html.Br(),
                dcc.Dropdown(
                    id='Product_importance',
                    options=[
                        {'label': 'low', 'value': 'low'},
                        {'label': 'medium', 'value': 'medium'},
                        {'label': 'high', 'value': 'high'}
                    ],
                    placeholder="Product Importance"
                ),
                html.Br(),
                dcc.Dropdown(
                    id='Gender',
                    options=[
                        {'label': 'Female', 'value': 'F'},
                        {'label': 'Male', 'value': 'M'}
                    ],
                    placeholder="Gender"
                ),
                html.Br(),
                dcc.Input(
                    id="Discount_offered",
                    placeholder="Discount (%)",
                    type="number",
                    style={'width': '100%'}
                ),
                html.Br(),
                html.Br(),
                dbc.Button(
                    id="button",
                    n_clicks=0,
                    children="Submit",
                    color="primary"
                ),
                html.Br(),
                html.Br(),
                html.Br(),
                dcc.Markdown(
                    "Prediction (0 - Predicted Late, 1 - Predicted On Time)"),
                html.Div(id="prediction"),
                html.Br(),
                dcc.Markdown("Probability of Late Shipment"),
                html.Div(id="predict_proba")
            ], md=4),
            ]),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
])


# ----------------------------------------------------------------
# Connect the Plotly graphs with Dash Components


@app.callback(
    Output(component_id='scatter', component_property='figure'),
    [Input(component_id='feature', component_property='value')]
)
def create_scatter(feature):
    """
    Creates scatter plot based on feature chosen by the user.
    Plots chosen feature on x-axis and target variable on y-axis
    Adds a line of best fit
    """
    if feature is None:
        raise PreventUpdate
    else:
        feature_list = X_test[feature].values.tolist()
        proba = model.predict_proba(X_test)

        late_proba = proba[:, :1]
        proba_list = [late_proba[i][0] for i in range(len(late_proba))]

        data = {feature: feature_list, "late_shipment_probability": proba_list}
        df_proba = pd.DataFrame(data=data)

        fig = px.scatter(
            df_proba,
            x=df_proba[feature],
            y=df_proba["late_shipment_probability"],
            trendline="ols",
            size_max=30)
        fig.update_layout(title=feature + ' Scatter Plot')

    return fig


@app.callback(
    Output(component_id='histogram_graph', component_property='figure'),
    [Input(component_id='bar_feature', component_property='value')]
)
def create_histogram(bar_feature):
    """
    Takes the continous feature chosen by the user and plots the
    distribution of the feature's class values in a bar chart
    """
    if bar_feature is None:
        raise PreventUpdate
    else:
        series = df[bar_feature].value_counts()
        series = series.sort_index()

        categories = list(series.index)
        values = list(series.values)

        fig = px.bar(df, x=categories, y=values,
                        title=bar_feature + " Distribution")
        fig.update_layout(xaxis_title=bar_feature, yaxis_title='Number of Shipments')
    return fig


@app.callback(
    Output(component_id='pie_chart', component_property='figure'),
    [Input(component_id='pie_feature', component_property='value')]
)
def create_pie_chart(pie_feature):
    """
    Takes the categorical feature chosen by the user and plots the
    distribution of the feature's class values in a pie chart
    """
    if pie_feature is None:
        raise PreventUpdate
    else:
        series = df[pie_feature].value_counts()

        categories = list(series.index)
        values = list(series.values)

        fig = px.pie(df, values=values, names=categories,
                        title=pie_feature + " Class Proportions")

        fig.update_traces(hole=.4, hoverinfo="label+percent+name")
    return fig


@app.callback(
    [Output(component_id='prediction', component_property='children'),
     Output(component_id='predict_proba', component_property='children')],
    [Input(component_id='button', component_property='n_clicks')],
    [State("Warehouse_block", "value"),
     State("Mode_of_Shipment", "value"),
     State("Customer_care_calls", "value"),
     State("Cost_of_the_Product", "value"),
     State("Prior_purchases", "value"),
     State("Product_importance", "value"),
     State("Gender", "value"),
     State("Discount_offered", "value"),
     State("Weight_in_gms", "value")]
)
def predict_late_shipment(
        n_clicks,
        Warehouse_block,
        Mode_of_Shipment,
        Customer_care_calls,
        Cost_of_the_Product,
        Prior_purchases,
        Product_importance,
        Gender,
        Discount_offered,
        Weight_in_gms):
    """
    Predicts if the data input by the user about a shipment will
    arrive late. Also provides probability of shipment arriving
    late.
    """
    if n_clicks == 0:
        raise PreventUpdate
    else:
        input_arr = np.array([[Warehouse_block,
                               Mode_of_Shipment,
                               Customer_care_calls,
                               Cost_of_the_Product,
                               Prior_purchases,
                               Product_importance,
                               Gender,
                               Discount_offered,
                               Weight_in_gms]])
        df_inp = pd.DataFrame(data=input_arr, columns=X_test.columns)
        prediction = model.predict(df_inp)
        predict_proba = str(
            round(
                model.predict_proba(df_inp)[0][0] * 100,
                2)) + "%"

    return prediction, predict_proba


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
