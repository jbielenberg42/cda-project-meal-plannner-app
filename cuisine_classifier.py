from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import json


class CuisineClassifier:

    def __init__(self, data_file_path):

        self.X, self.y = self._format_and_clean_ingredient_data(data_file_path)
        self.clf = self._train_classifier()
        self.features = self.X.columns
        self.cuisines = self.clf.classes_
        self.oob_predictions = self.cuisines[
            self.clf.oob_decision_function_.argmax(axis=1)
        ]
        self.classification_report = classification_report(self.y, self.oob_predictions)
        self.cuisine_pca = self._cuisine_pca()

    def _format_and_clean_ingredient_data(self, data_file_path):
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

    def _train_classifier(self):
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

    def cuisine_prediction_df(self, recipe_ingredient_matrix):
        df = pd.DataFrame()
        cuisine_probabilities = self.clf.predict_proba(recipe_ingredient_matrix)
        for i, cuisine in enumerate(self.clf.classes_):
            df[f"prob_{cuisine}"] = cuisine_probabilities[:, i]

        df["predicted_cuisine"] = self.clf.predict(recipe_ingredient_matrix)
        df["predicted_cuisine_prob"] = np.max(cuisine_probabilities, axis=1)
        return df

    def _cuisine_pca(self, explained_variance_perc=0.95):
        # Laziness in notation here. Self.X != X
        ingredient_counts = self.X.groupby(self.y).sum()
        X = ingredient_counts.div(ingredient_counts.sum(axis=1), axis=0)
        X_T = X.T
        U, S, _ = np.linalg.svd(X_T)
        n_components = np.argmax(np.cumsum(S / S.sum()) >= explained_variance_perc)
        vectors = pd.DataFrame(index=X.index)
        for i in range(n_components):
            vectors[f"{i+1}"] = (U[:, i] @ X_T) / (S[i] ** 0.05)
        return vectors
