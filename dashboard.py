from explainerdashboard import ExplainerDashboard, ExplainerHub, ClassifierExplainer, RegressionExplainer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from explainerdashboard.custom import *
from dash_bootstrap_components.themes import CYBORG, PULSE, DARKLY, FLATLY, CERULEAN, SKETCHY
import dash_bootstrap_components as dbc

import openpyxl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas import DataFrame

from dash import Dash, callback, html, dcc, dash_table, Input, Output, State, MATCH, ALL


##### Load data with sheet name #########


data = pd.read_excel('main_data.xlsx', sheet_name="GDP").dropna()
X_data = data.drop(['Rgr', 'Nogr'], axis=1)

y = data['Rgr']


#####################################
####### REAL GDP
#####################################

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y, test_size=0.2, random_state=1)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#model = RandomForestRegressor(n_estimators=50, max_depth=5)


model = RandomForestRegressor(n_estimators=400,
                              n_jobs=-1,
                              oob_score=True,
                              bootstrap=True,
                              max_depth=5,
                              random_state=42)

model.fit(X_train, y_train)


############ Make Dashboard ##########


explainer1 = RegressionExplainer(model, X_test, y_test)


class CustomModelTab1A(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Selected Drivers")
        self.importance = ImportancesComposite(explainer,
                                               title='Impact',
                                               hide_importances=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    self.importance.layout(),
                    html.H3(f"Clearly {self.explainer.columns_ranked_by_shap()[0]} was the most important"
                            f", followed by {self.explainer.columns_ranked_by_shap()[1]}"
                            f" and {self.explainer.columns_ranked_by_shap()[2]}.")
                ])
            ])
        ])


class CustomModelTab1B(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Model Performance")
        self.Reg_summary = RegressionModelStatsComposite(explainer,
                                                         title='Impact',
                                                         hide_predsvsactual=False, hide_residuals=False,
                                                         hide_regvscol=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    self.Reg_summary.layout(),


                ])
            ])
        ])


class CustomPredictionsTab1C(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Model Predictions")

        self.prediction = IndividualPredictionsComposite(explainer,
                                                         hide_predindexselector=False, hide_predictionsummary=False,
                                                         hide_contributiongraph=False, hide_pdp=False,
                                                         hide_contributiontable=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Individual Prediction:"),
                    self.prediction.layout()
                ])

            ])
        ])


class CustomPredictionsTab1D(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="What if Scenarios")

        self.what_if = WhatIfComposite(explainer,
                                       hide_whatifindexselector=False, hide_inputeditor=False,
                                       hide_whatifcontribution=False, hide_whatifpdp=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Individual Prediction:"),
                    self.what_if.layout()
                ])

            ])
        ])


class CustomPredictionsTab1E(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="SHAP Dependencies")

        self.shap_depend = ShapDependenceComposite(explainer,
                                                   hide_shapsummary=False, hide_shapdependence=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("SHAP Dependencies:"),
                    self.shap_depend.layout()
                ])

            ])
        ])


class CustomPredictionsTab1F(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Interacting Features")

        self.interaction = ShapInteractionsComposite(explainer, 
                                                      hide_interactionsummary=False, 
                                                      hide_interactiondependence=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Interacting Features:"),
                    self.interaction.layout()
                ])

            ])
        ])


db1 = ExplainerDashboard(explainer1, [CustomModelTab1A, CustomModelTab1B, CustomPredictionsTab1C,
                                      CustomPredictionsTab1D, CustomPredictionsTab1E, CustomPredictionsTab1F],
                         title='Real GDP Growth Rate Prediction for Nigeria', header_hide_selector=False,
                         bootstrap=CYBORG)


#############################################
# NON OIL REAL GDP
#############################################


data2 = pd.read_excel('main_data.xlsx', sheet_name="GDP").dropna()
X2_data = data2.drop(['Rgr', 'Nogr'], axis=1)

y2 = data2['Nogr']


X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2_data, y2, test_size=0.2, random_state=1)
print(X2_train.shape, y2_train.shape, X2_test.shape, y2_test.shape)

#model = RandomForestRegressor(n_estimators=50, max_depth=5)


model2 = RandomForestRegressor(n_estimators=400,
                               n_jobs=-1,
                               oob_score=True,
                               bootstrap=True,
                               max_depth=5,
                               random_state=42)

model2.fit(X2_train, y2_train)


############ Make Dashboard ##########


explainer2 = RegressionExplainer(model2, X2_test, y2_test)


class CustomModelTab2A(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Selected Drivers")
        self.importance = ImportancesComposite(explainer,
                                               title='Impact',
                                               hide_importances=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    self.importance.layout(),
                    html.H3(f"Clearly {self.explainer.columns_ranked_by_shap()[0]} was the most important"
                            f", followed by {self.explainer.columns_ranked_by_shap()[1]}"
                            f" and {self.explainer.columns_ranked_by_shap()[2]}.")
                ])
            ])
        ])


class CustomModelTab2B(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Model Performance")
        self.Reg_summary = RegressionModelStatsComposite(explainer,
                                                         title='Impact',
                                                         hide_predsvsactual=False, hide_residuals=False,
                                                         hide_regvscol=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    self.Reg_summary.layout(),


                ])
            ])
        ])


class CustomPredictionsTab2C(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title=" Model Predictions")

        self.prediction = IndividualPredictionsComposite(explainer,
                                                         hide_predindexselector=False, hide_predictionsummary=False,
                                                         hide_contributiongraph=False, hide_pdp=False,
                                                         hide_contributiontable=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Individual Prediction:"),
                    self.prediction.layout()
                ])

            ])
        ])


class CustomPredictionsTab2D(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="What if Scenarios")

        self.what_if = WhatIfComposite(explainer,
                                       hide_whatifindexselector=False, hide_inputeditor=False,
                                       hide_whatifcontribution=False, hide_whatifpdp=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Individual Prediction:"),
                    self.what_if.layout()
                ])

            ])
        ])


class CustomPredictionsTab2E(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="SHAP Dependencies")

        self.shap_depend = ShapDependenceComposite(explainer,
                                                   hide_shapsummary=False, hide_shapdependence=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("SHAP Dependencies:"),
                    self.shap_depend.layout()
                ])

            ])
        ])


class CustomPredictionsTab2F(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Interacting Features")

        self.interaction = ShapInteractionsComposite(explainer, 
                                                      hide_interactionsummary=False, 
                                                      hide_interactiondependence=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Interacting Features:"),
                    self.interaction.layout()
                ])

            ])
        ])


db2 = ExplainerDashboard(explainer2, [CustomModelTab2A, CustomModelTab2B, CustomPredictionsTab2C,
                                      CustomPredictionsTab2D, CustomPredictionsTab2E, CustomPredictionsTab2F],
                         title='Nonoil Growth Rate Prediction for Nigeria', header_hide_selector=False,
                         bootstrap=CYBORG)


###########################################
# INFLATION
############################################


data3 = pd.read_excel('main_data.xlsx', sheet_name="inflation").dropna()

X3_data = data3.drop(['inflation'], axis=1)
y3 = data3['inflation']


X3_train, X3_test, y3_train, y3_test = train_test_split(
    X3_data, y3, test_size=0.2, random_state=1)
print(X3_train.shape, y3_train.shape, X3_test.shape, y3_test.shape)

#model = RandomForestRegressor(n_estimators=50, max_depth=5)


model3 = RandomForestRegressor(n_estimators=400,
                               n_jobs=-1,
                               oob_score=True,
                               bootstrap=True,
                               max_depth=5,
                               random_state=42)

model3.fit(X3_train, y3_train)


############ Make Dashboard ##########


explainer3 = RegressionExplainer(model3, X3_test, y3_test)


class CustomModelTab3A(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Selected Drivers")
        self.importance = ImportancesComposite(explainer,
                                               title='Impact',
                                               hide_importances=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    self.importance.layout(),
                    html.H3(f"Clearly {self.explainer.columns_ranked_by_shap()[0]} was the most important"
                            f", followed by {self.explainer.columns_ranked_by_shap()[1]}"
                            f" and {self.explainer.columns_ranked_by_shap()[2]}.")
                ])
            ])
        ])


class CustomModelTab3B(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Model Performance")
        self.Reg_summary = RegressionModelStatsComposite(explainer,
                                                         title='Impact',
                                                         hide_predsvsactual=False, hide_residuals=False,
                                                         hide_regvscol=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    self.Reg_summary.layout(),


                ])
            ])
        ])


class CustomPredictionsTab3C(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Model Predictions")

        self.prediction = IndividualPredictionsComposite(explainer,
                                                         hide_predindexselector=False, hide_predictionsummary=False,
                                                         hide_contributiongraph=False, hide_pdp=False,
                                                         hide_contributiontable=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Individual Prediction:"),
                    self.prediction.layout()
                ])

            ])
        ])


class CustomPredictionsTab3D(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="What if Scenarios")

        self.what_if = WhatIfComposite(explainer,
                                       hide_whatifindexselector=False, hide_inputeditor=False,
                                       hide_whatifcontribution=False, hide_whatifpdp=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Individual Prediction:"),
                    self.what_if.layout()
                ])

            ])
        ])


class CustomPredictionsTab3E(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="SHAP Dependencies")

        self.shap_depend = ShapDependenceComposite(explainer,
                                                   hide_shapsummary=False, hide_shapdependence=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("SHAP Dependencies:"),
                    self.shap_depend.layout()
                ])

            ])
        ])


class CustomPredictionsTab3F(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Interacting Features")

        self.interaction = ShapInteractionsComposite(explainer, 
                                                      hide_interactionsummary=False, 
                                                      hide_interactiondependence=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Interacting Features:"),
                    self.interaction.layout()
                ])

            ])
        ])


db3 = ExplainerDashboard(explainer3, [CustomModelTab3A, CustomModelTab3B, CustomPredictionsTab3C,
                                      CustomPredictionsTab3C, CustomPredictionsTab3E, CustomPredictionsTab3F],
                         title='Inflation Prediction for Nigeria', header_hide_selector=False,
                         bootstrap=CYBORG)



####################################################################
#####               INFLATION
####################################################################

data = pd.read_excel('Elijah_inflation.xlsx', sheet_name="Full Sample").dropna()
X_data = data.drop(['Headline', 'Date', 'Core'], axis=1)

y = data['Headline']


#####################################
###    Full Sample Regime 
#####################################

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y, test_size=0.2, random_state=1)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#model = RandomForestRegressor(n_estimators=50, max_depth=5)


model = RandomForestRegressor(n_estimators=400,
                              n_jobs=-1,
                              oob_score=True,
                              bootstrap=True,
                              max_depth=5,
                              random_state=42)

model.fit(X_train, y_train)


############ Make Dashboard ##########


explainer4 = RegressionExplainer(model, X_test, y_test)


class CustomModelTab4A(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Selected Drivers")
        self.importance = ImportancesComposite(explainer,
                                               title='Impact',
                                               hide_importances=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    self.importance.layout(),
                    html.H3(f"Clearly {self.explainer.columns_ranked_by_shap()[0]} was the most important"
                            f", followed by {self.explainer.columns_ranked_by_shap()[1]}"
                            f" and {self.explainer.columns_ranked_by_shap()[2]}.")
                ])
            ])
        ])


class CustomModelTab4B(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Model Performance")
        self.Reg_summary = RegressionModelStatsComposite(explainer,
                                                         title='Impact',
                                                         hide_predsvsactual=False, hide_residuals=False,
                                                         hide_regvscol=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    self.Reg_summary.layout(),


                ])
            ])
        ])


class CustomPredictionsTab4C(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Model Predictions")

        self.prediction = IndividualPredictionsComposite(explainer,
                                                         hide_predindexselector=False, hide_predictionsummary=False,
                                                         hide_contributiongraph=False, hide_pdp=False,
                                                         hide_contributiontable=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Prediction:"),
                    self.prediction.layout()
                ])

            ])
        ])


class CustomPredictionsTab4D(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="What if Scenarios")

        self.what_if = WhatIfComposite(explainer,
                                       hide_whatifindexselector=False, hide_inputeditor=False,
                                       hide_whatifcontribution=False, hide_whatifpdp=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Scenarios:"),
                    self.what_if.layout()
                ])

            ])
        ])


class CustomPredictionsTab4E(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="SHAP Dependencies")

        self.shap_depend = ShapDependenceComposite(explainer,
                                                   hide_shapsummary=False, hide_shapdependence=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("SHAP Dependencies:"),
                    self.shap_depend.layout()
                ])

            ])
        ])


class CustomPredictionsTab4F(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Interacting Features")

        self.interaction = ShapInteractionsComposite(explainer, 
                                                      hide_interactionsummary=False, 
                                                      hide_interactiondependence=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Interacting Features:"),
                    self.interaction.layout()
                ])

            ])
        ])

db4 = ExplainerDashboard(explainer4, [CustomModelTab4A, CustomModelTab4B, CustomPredictionsTab4C,
                                      CustomPredictionsTab4D, CustomPredictionsTab4E, CustomPredictionsTab4F],
                         title='Full Inflation Regime', header_hide_selector=False,
                         bootstrap=CYBORG)


#############################################
######     Low Inflation Regime 
#############################################


data2 = pd.read_excel('Elijah_inflation.xlsx', sheet_name="Low Inflation Regime").dropna()
X2_data = data2.drop(['Headline', 'Date', 'Core'], axis=1)

y2 = data2['Headline']


X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2_data, y2, test_size=0.2, random_state=1)
print(X2_train.shape, y2_train.shape, X2_test.shape, y2_test.shape)

#model = RandomForestRegressor(n_estimators=50, max_depth=5)


model2 = RandomForestRegressor(n_estimators=400,
                               n_jobs=-1,
                               oob_score=True,
                               bootstrap=True,
                               max_depth=5,
                               random_state=42)

model2.fit(X2_train, y2_train)


############ Make Dashboard ##########


explainer5 = RegressionExplainer(model2, X2_test, y2_test)


class CustomModelTab5A(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Selected Drivers")
        self.importance = ImportancesComposite(explainer,
                                               title='Impact',
                                               hide_importances=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    self.importance.layout(),
                    html.H3(f"Clearly {self.explainer.columns_ranked_by_shap()[0]} was the most important"
                            f", followed by {self.explainer.columns_ranked_by_shap()[1]}"
                            f" and {self.explainer.columns_ranked_by_shap()[2]}.")
                ])
            ])
        ])


class CustomModelTab5B(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Model Performance")
        self.Reg_summary = RegressionModelStatsComposite(explainer,
                                                         title='Impact',
                                                         hide_predsvsactual=False, hide_residuals=False,
                                                         hide_regvscol=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    self.Reg_summary.layout(),


                ])
            ])
        ])


class CustomPredictionsTab5C(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Model Predictions")

        self.prediction = IndividualPredictionsComposite(explainer,
                                                         hide_predindexselector=False, hide_predictionsummary=False,
                                                         hide_contributiongraph=False, hide_pdp=False,
                                                         hide_contributiontable=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Individual Prediction:"),
                    self.prediction.layout()
                ])

            ])
        ])


class CustomPredictionsTab5D(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="What if Scenarios")

        self.what_if = WhatIfComposite(explainer,
                                       hide_whatifindexselector=False, hide_inputeditor=False,
                                       hide_whatifcontribution=False, hide_whatifpdp=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Individual Prediction:"),
                    self.what_if.layout()
                ])

            ])
        ])


class CustomPredictionsTab5E(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="SHAP Dependencies")

        self.shap_depend = ShapDependenceComposite(explainer,
                                                   hide_shapsummary=False, hide_shapdependence=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("SHAP Dependencies:"),
                    self.shap_depend.layout()
                ])

            ])
        ])


class CustomPredictionsTab5F(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Interacting Features")

        self.interaction = ShapInteractionsComposite(explainer, 
                                                      hide_interactionsummary=False, 
                                                      hide_interactiondependence=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Interacting Features:"),
                    self.interaction.layout()
                ])

            ])
        ])

db5 = ExplainerDashboard(explainer5, [CustomModelTab5A, CustomModelTab5B, CustomPredictionsTab5C,
                                      CustomPredictionsTab5D, CustomPredictionsTab5E, CustomPredictionsTab5F],
                         title='Low Inlfation Regime', header_hide_selector=False,
                         bootstrap=CYBORG)


###########################################
# High Inflation Regime
############################################


data3 = pd.read_excel('Elijah_inflation.xlsx', sheet_name="High Inflation Regime").dropna()

X3_data = data3.drop(['Headline', 'Date', 'Core'], axis=1)
y3 = data3['Headline']


X3_train, X3_test, y3_train, y3_test = train_test_split(
    X3_data, y3, test_size=0.2, random_state=1)
print(X3_train.shape, y3_train.shape, X3_test.shape, y3_test.shape)

#model = RandomForestRegressor(n_estimators=50, max_depth=5)


model3 = RandomForestRegressor(n_estimators=400,
                               n_jobs=-1,
                               oob_score=True,
                               bootstrap=True,
                               max_depth=5,
                               random_state=42)

model3.fit(X3_train, y3_train)


############ Make Dashboard ##########


explainer6 = RegressionExplainer(model3, X3_test, y3_test)


class CustomModelTab6A(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Selected Drivers")
        self.importance = ImportancesComposite(explainer,
                                               title='Impact',
                                               hide_importances=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    self.importance.layout(),
                    html.H3(f"Clearly {self.explainer.columns_ranked_by_shap()[0]} was the most important"
                            f", followed by {self.explainer.columns_ranked_by_shap()[1]}"
                            f" and {self.explainer.columns_ranked_by_shap()[2]}.")
                ])
            ])
        ])


class CustomModelTab6B(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Model Performance")
        self.Reg_summary = RegressionModelStatsComposite(explainer,
                                                         title='Impact',
                                                         hide_predsvsactual=False, hide_residuals=False,
                                                         hide_regvscol=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    self.Reg_summary.layout(),


                ])
            ])
        ])


class CustomPredictionsTab6C(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Model Predictions")

        self.prediction = IndividualPredictionsComposite(explainer,
                                                         hide_predindexselector=False, hide_predictionsummary=False,
                                                         hide_contributiongraph=False, hide_pdp=False,
                                                         hide_contributiontable=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Individual Prediction:"),
                    self.prediction.layout()
                ])

            ])
        ])


class CustomPredictionsTab6D(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="What if Scenarios")

        self.what_if = WhatIfComposite(explainer,
                                       hide_whatifindexselector=False, hide_inputeditor=False,
                                       hide_whatifcontribution=False, hide_whatifpdp=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Individual Prediction:"),
                    self.what_if.layout()
                ])

            ])
        ])


class CustomPredictionsTab6E(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="SHAP Dependencies")

        self.shap_depend = ShapDependenceComposite(explainer,
                                                   hide_shapsummary=False, hide_shapdependence=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("SHAP Dependencies:"),
                    self.shap_depend.layout()
                ])

            ])
        ])


class CustomPredictionsTab6F(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Interacting Features")

        self.interaction = ShapInteractionsComposite(explainer, 
                                                      hide_interactionsummary=False, 
                                                      hide_interactiondependence=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Interacting Features:"),
                    self.interaction.layout()
                ])

            ])
        ])


db6 = ExplainerDashboard(explainer6, [CustomModelTab6A, CustomModelTab6B, CustomPredictionsTab6C,
                                      CustomPredictionsTab6D, CustomPredictionsTab6E, CustomPredictionsTab6F],
                         title='High Inflation Regime', name="db2",
            description="This is model option two",  header_hide_selector=False,
                         bootstrap=CYBORG)

hub=ExplainerHub([db1, db2, db3, db4, db5, db6], title="Model Comparison",
            description="This explainer dashboard hub shows different macroeconomic dashboards, each dashboard contains predictions of a trained machine learning model explainable.",
            bootstrap=CERULEAN)

hub.to_yaml("hub.yaml")
hub2 = ExplainerHub.from_config("hub.yaml")

app=hub2.run(host='0.0.0.0', port=8050, use_waitress=True)








