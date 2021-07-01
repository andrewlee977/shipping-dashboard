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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, y_test


def perm_imp(model, X_test, y_test):
    perm_imp = permutation_importance(model, X_test, y_test, n_repeats=10, n_jobs=-1, random_state=42)
    data = {"imp_mean": perm_imp["importances_mean"],
            "imp_std": perm_imp["importances_std"]}
    df_perm = pd.DataFrame(data, index=X_test.columns).sort_values("imp_mean")
    # df_perm["imp_mean"].tail(10).plot(kind="barh")
    return df_perm

external_stylesheets = [
    dbc.themes.JOURNAL, # Bootswatch theme
    'https://use.fontawesome.com/releases/v5.9.0/css/all.css', # for social media icons
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# ---------- 

# Load DataFrame
df = load_data("dash_ready_data.csv")

# Load Model
model = load_model("gbc_gs.pk1")

X_test, y_test = split_data(df, model)

df_perm = perm_imp(model, X_test, y_test)


# def predict_


# Permutation Importance Graph
def permutation_graph():
    fig = go.Figure(go.Bar(
                x=df_perm["imp_mean"],
                y=df_perm.index,
                orientation='h'))

    fig.update_layout(title={"text": "Permutation Importance",
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
                        colorscale="Viridis"

    ))
    # z = [[111, 784], [725, 580]]
    # fig = ff.create_annotated_heatmap(z)
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

# fig2
# fig2 = px.scatter(df, x="Weight_in_gms", y="Reached_on_time", size_max=30)

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.Div([

        html.H1("Shipping Dashboard", style={'text-align': 'center'}),

        html.Br(),

        dbc.Row([
            dbc.Col([
                dcc.Graph(id='perm_imp', figure=permutation_graph()),
            ]),
            dbc.Col([
                dcc.Graph(id='confusion', figure=confusion_matrix()),
            ]),
        ]),

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

        # html.Div([
        #     dcc.Markdown("Hello World")
        # ]),

        dcc.Graph(id='scatter', figure={}),

        
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
            placeholder="# customer care calls",
            type="number"
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
            value=3
        ),
        html.Br(),
        dcc.Input(
            id="Cost_of_the_Product",
            placeholder="Product Cost",
            type="number"
        ),
        html.Br(),
        html.Br(),
        dcc.Input(
            id="Prior_purchases",
            placeholder="# Prior Purchases",
            type="number"
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
            type="number"
        ),
        html.Br(),
        html.Br(),
        dcc.Input(
            id="Weight_in_gms",
            placeholder="Product Weight (grams)",
            type="number"
        ),
        html.Br(),
        dbc.Button(
                id="button",
                n_clicks=0,
                children="Submit",
                color="primary"
        ),
        html.Br(),
        html.Div(id="prediction"),
        html.Div(id="predict_proba")

    ])


])


# ------------------------------------------------------------------------------


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

        fig = px.scatter(df_proba, x=df_proba[feature], y=df_proba["late_shipment_probability"], size_max=30)
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
        predict_proba = model.predict_proba(df_inp)

    return prediction, predict_proba[0]



# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
    
    