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
    return pd.read_csv(filepath, index_col="ID")


def load_model(filepath):
    return pickle.load(open(filepath, 'rb'))


def split(df):
    X = df.drop(columns="Reached_on_time")
    y = df["Reached_on_time"]
    return X, y


def split_data(df, model):
    X, y = split(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    return X_test, y_test


def perm_imp(model, X_test, y_test):
    perm_imp = permutation_importance(model, X_test, y_test, n_repeats=10,
                                      n_jobs=-1, random_state=42)
    data = {"imp_mean": perm_imp["importances_mean"],
            "imp_std": perm_imp["importances_std"]}
    df_perm = pd.DataFrame(data, index=X_test.columns).sort_values("imp_mean")
    # df_perm["imp_mean"].tail(10).plot(kind="barh")
    return df_perm

external_stylesheets = [
    dbc.themes.JOURNAL, # Bootswatch theme
    'https://use.fontawesome.com/releases/v5.9.0/css/all.css',
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# ---------- 

# Load DataFrame
df = load_data("../data/dash_ready_data.csv")

# Load Model
model = load_model("../model/gbc_gs.pk1")

X_test, y_test = split_data(df, model)

df_perm = perm_imp(model, X_test, y_test)


# Permutation Importance Graph
def permutation_graph():
    fig = go.Figure(go.Bar(
                x=df_perm["imp_mean"],
                y=df_perm.index,
                orientation='h'))

    fig.update_layout(title={"text": "Permutation Feature Importance",
                            'y':0.9,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'},
                    xaxis_title="Model Accuracy Decrease",
                    yaxis_title="Features",
                    width=1000,
                    height=500)

    return fig

def confusion_matrix():
    fig = go.Figure(data=go.Heatmap(
                        z=[[111, 784], [725, 580]],
                        x=["Shipment Late", "Shipment On Time"],
                        y=["Shipment late", "Shipment On Time"],
                        colorscale="Viridis"))
    fig.update_layout(title={"text": "Confusion Matrix",
                            'y':0.9,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'},
                    xaxis_title="Predicted Label",
                    yaxis_title="True Label",
                    width=600,
                    height=500)
    return fig


# ----------------------------------------------------------------------
# App layout
app.layout = html.Div([
    html.Div([
        html.H1("Shipping Dashboard", style={'text-align': 'center'}),
        html.Br(),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='perm_imp', figure=permutation_graph()),
                html.Div([
                    dcc.Graph(id='confusion', figure=confusion_matrix())
                ]),
                
            ]),
            dbc.Col([
                html.Div("""ABOUT THIS DASHBOARD""", 
                    style={'color': 'black', 'fontSize': 30}),
                dcc.Link(href=
                        "https://www.kaggle.com/prachi13/customer-analytics"),
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
            ]),
        ]),
        html.Div("Scatter Plot", style={'color': 'black', 'fontSize': 30}),
        html.Br(),
        dcc.Dropdown(id="feature",
                    options=[
                        {"label": "Weight_in_gms", "value": "Weight_in_gms"},
                        {"label": "Discount_offered", "value": "Discount_offered"},
                        {"label": "Prior_purchases", "value": "Prior_purchases"},
                        {"label": "Customer_care_calls", "value": "Customer_care_calls"},
                        {"label": "Cost_of_the_Product", "value": "Cost_of_the_Product"},
                        {"label": "Warehouse_block", "value": "Warehouse_block"},
                        {"label": "Customer_rating", "value": "Customer_rating"},
                        {"label": "Mode_of_Shipment", "value": "Mode_of_Shipment"},
                        {"label": "Product_importance", "value": "Product_importance"},
                        {"label": "Gender", "value": "Gender"}],
                    placeholder="Choose Feature",
                    style={'width': "40%"}
                    ),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='scatter', figure={}),
            ]),
            dbc.Col([
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
                dcc.Markdown("Customer Rating"),
                dcc.Slider(
                    id="Customer_rating",
                    min=1,
                    max=5,
                    step=1,
                    marks={
                        1: "1",
                        2: "2",
                        3: "3",
                        4: "4",
                        5: "5"
                    },
                ),
                html.Br(),
                dcc.Input(
                    id="Cost_of_the_Product",
                    placeholder="Product Cost",
                    type="number",
                    style={'width': '100%'}
                ),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                dbc.Button(
                        id="button",
                        n_clicks=0,
                        children="Submit",
                        color="primary"
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
                html.Br(),
                dcc.Input(
                    id="Discount_offered",
                    placeholder="Discount (%)",
                    type="number",
                    style={'width': '100%'}
                ),
                html.Br(),
                html.Br(),
                html.Br(),
                dcc.Input(
                    id="Weight_in_gms",
                    placeholder="Product Weight (Grams)",
                    type="number",
                    style={'width': '100%'}
                ),
                html.Br(),
                html.Br(),
                html.Br(),
                dcc.Markdown("Prediction (0 - Predicted Late, 1 - Predicted On Time)"),
                html.Div(id="prediction"),
                html.Br(),
                dcc.Markdown("Probability of Late Shipment"),
                html.Div(id="predict_proba")
            ], md=4),
        ]),
    ])
])

# ----------------------------------------------------------------

# Connect the Plotly graphs with Dash Components
@app.callback(
     Output(component_id='scatter', component_property='figure'),
    [Input(component_id='feature', component_property='value')]
)


def create_scatter(feature):
    if feature is None:
        raise PreventUpdate
    else:
        feature_list = X_test[feature].values.tolist()
        proba = model.predict_proba(X_test)

        proba_data = proba[:, :1]
        proba_list = [proba_data[i][0] for i in range(len(proba_data))]

        data = {f"{feature}": feature_list, "late_shipment_probability": proba_list}
        df_proba = pd.DataFrame(data=data)

        fig = px.scatter(df_proba, x=df_proba[feature], y=df_proba["late_shipment_probability"],
                         trendline="ols", size_max=30)
        fig.update_layout(title=f'{feature} Scatter Plot')

    return fig


@app.callback(
    [Output(component_id='prediction', component_property='children'),
     Output(component_id='predict_proba', component_property='children')],
    [Input(component_id='button', component_property='n_clicks')],
    [State("Warehouse_block", "value"),
     State("Mode_of_Shipment", "value"),
     State("Customer_care_calls", "value"),
     State("Customer_rating", "value"),
     State("Cost_of_the_Product", "value"),
     State("Prior_purchases", "value"),
     State("Product_importance", "value"),
     State("Gender", "value"),
     State("Discount_offered", "value"),
     State("Weight_in_gms", "value")]
)


def predict_late_shipment(n_clicks, Warehouse_block, Mode_of_Shipment,
                          Customer_care_calls, Customer_rating,
                          Cost_of_the_Product, Prior_purchases,
                          Product_importance, Gender, Discount_offered,
                          Weight_in_gms):
    if n_clicks == 0:
        raise PreventUpdate
    else:
        input_arr = np.array([[Warehouse_block, Mode_of_Shipment,
                               Customer_care_calls, Customer_rating,
                               Cost_of_the_Product, Prior_purchases,
                               Product_importance, Gender, Discount_offered,
                               Weight_in_gms]])
        df_inp = pd.DataFrame(data=input_arr, columns=X_test.columns)
        prediction = model.predict(df_inp)
        predict_proba = str(round(model.predict_proba(df_inp)[0][0] * 100, 2)) + "%"
        
    return prediction, predict_proba

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
    