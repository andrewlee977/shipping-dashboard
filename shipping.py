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
                            shipping data from Kaggle. Below are some
                            important insights from exploring the data as well
                            as a predictor at the bottom of the webpage. The
                            dataset contains data from an electronic products
                            company. The target variable is `Reached_on_time`,
                            which is a binary value representing if a product
                            arrived to the customer on time."""),

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
        html.Div("Confusion Matrix",
                 style={'color': 'black', 'fontSize': 20}),
        dcc.Markdown("""The confusion matrix table indicates
                             the performance of a classification model.
                             Our confusion matrix tells us that our model
                             has a high Recall Score (88%) but low Precision
                             Score (57%) with an Accuracy Score of 69%
                             (starting from 50% baseline). """),
        html.Div("Scatter Plot",
                 style={'color': 'black', 'fontSize': 20}),
        dcc.Markdown("""The scatter plot illustrates the relationship
                             between the chosen feature and the target variable
                             `Reached_on_time`. Note that beyond `Weight_in_gms`
                             and `Discount_offered`, a significant relationship
                             is not found."""),
        html.Div("Predict Late Shipping",
                 style={'color': 'black', 'fontSize': 20}),
        dcc.Markdown("""Use the interactive dashboard to predict
                             if a product will arrive late. Output: (0 - Predicted
                             Late, 1 = Predicted On Time) and the probability
                             of the shipment being late."""),
        html.Div("Tools",
                 style={'color': 'black', 'fontSize': 20}),
        dcc.Markdown("This dashboard was built using Dash by Plotly."),
        dcc.Markdown("""The Gradient Boosting Classifier model is from
                             Sci-kit Learn's library."""),
        dcc.Markdown("This app is being served on an AWS EC2 instance."),

        html.Div("Improvements",
                 style={'color': 'black', 'fontSize': 20}),
        dcc.Markdown("""In order to improve this dashboard, I would
                             first work on improving the model. Although
                             I delivered the model after a round of tuning
                             using Sci-kit Learn's GridSearch, the model
                             is currently held back by noise from most of
                             the existing features in the dataset. I would
                             simplify the model by cutting out all but the
                             top 3 relevant features. I believe this would
                             improve the model drastically. For the purpose of this
                             project, I decided to leave these irrelevant
                             features in for the sake of interactivity."""),
        dcc.Markdown("""I would also improve the UI/UX. I acknowledge
                             this isn't the best looking dashboard but is a
                             MVP that would be delivered to a stakeholder
                             such as a supply chain/logistics manager. This
                             is the product of a week's worth of work, so
                             improvements can definitely be made, and more
                             visualizations would be great for a more comprehensive
                             analysis of the shipping data. Thanks for
                             taking the time to explore!"""),
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
                    "Histogram",
                    style={
                        'color': 'black',
                        'fontSize': 30}),
                html.Br(),
                dcc.Dropdown(id="histogram_feature",
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
                             style={'width': "40%"}
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
                                  "value": "Cost_of_the_Product"},
                                 {"label": "Warehouse_block",
                                     "value": "Warehouse_block"},
                                 {"label": "Mode_of_Shipment",
                                     "value": "Mode_of_Shipment"},
                                 {"label": "Product_importance",
                                  "value": "Product_importance"},
                                 {"label": "Gender", "value": "Gender"}],
                             placeholder="Choose Feature",
                             style={'width': "40%"}
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
                             style={'width': "40%"}
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
        fig = px.scatter(
            df,
            x=[0],
            y=[0],
            trendline="ols",
            size_max=30)
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
        fig.update_layout(title=feature + 'Scatter Plot')

    return fig


@app.callback(
    Output(component_id='histogram_graph', component_property='figure'),
    [Input(component_id='histogram_feature', component_property='value')]
)
def create_histogram(histogram_feature):
    """
    Takes the continous feature chosen by the user and plots the
    distribution of the feature's class values in a bar chart
    """
    if histogram_feature is None:
        fig = px.bar(df, x=[0], y=[0],
                     title="Plot Continuous Features")
    else:
        series = df[histogram_feature].value_counts()
        series = series.sort_index()

        categories = list(series.index)
        values = list(series.values)

        fig = px.bar(df, x=categories, y=values,
                     title=histogram_feature + " Distribution")

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
        fig = px.pie(df, values=[0, 0], names=[" ", " "],
                     title="Categorical Features Proportion")
    else:
        series = df[pie_feature].value_counts()

        categories = list(series.index)
        values = list(series.values)

        fig = px.pie(df, values=values, names=categories,
                     title=pie_feature + " Class Percentage")

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
