from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import json


class CuisineClassifier:

    def __init__(self, data_file_path):

        self.X, self.y = self.format_and_clean_ingredient_data(data_file_path)
        self.clf = self.train_classifier()
        self.features = self.X.columns
        self.cuisines = self.clf.classes_
        self.oob_predictions = self.clf.oob_decision_function_.argmax(axis=1)
        self.classification_report = classification_report(self.y, self.oob_predictions)


    def format_and_clean_ingredient_data(self, data_file_path):
        # Load data
        with open(data_file_path, "r") as json_f:
            df = pd.DataFrame(json.load(json_f))

        #  Pivot the array column to one col per ingredient
        mlb = MultiLabelBinarizer()
        X = pd.DataFrame(mlb.fit_transform(df["ingredients"]), columns=mlb.classes_)
        y = df["cuisine"]

        # Remove infrequently used ingredients
        min_num_ingredient_occurences = 10
        X = X.loc[:, (X.sum(axis=0) >= min_num_ingredient_occurences)]

        # Fit an initial model for extracting feature importance
        rf_model = RandomForestClassifier(
            n_estimators=100, max_features=10, n_jobs=8, min_samples_split=50
        )
        rf_model.fit(X, y)
        feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
        feature_importance.sort_values(ascending=True, inplace=True)
        importance_threshold = 0.95
        important_features = feature_importance[
            feature_importance.cumsum() <= importance_threshold
        ]
        X = X[important_features.index]
        return X, y

    def train_classifier(self):
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_features=10,
            n_jobs=8,
            min_samples_split=50,
            oob_score=True,
        )
        rf_model.fit(self.X, self.y)
        return rf_model

    def extract_features_from_new_recipes(self, recipe_ingredients: pd.Series):
        n = len(self.features)
        m = len(recipe_ingredients)
        recipe_ingredients_mat = np.zeros((m, n))
        for i, feature in enumerate(self.features):
            recipe_ingredients_mat[:, i] = recipe_ingredients.apply(
                lambda x: 1 if feature in x else 0
            )
        return pd.DataFrame(recipe_ingredients_mat, columns=self.features)
