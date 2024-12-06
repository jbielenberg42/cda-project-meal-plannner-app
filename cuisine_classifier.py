from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import json


class CuisineClassifier:

    def __init__(self, data_file_path):

        self.X, self.y = self._format_and_clean_ingredient_data(data_file_path)
        self.clf = self._train_classifier()
        self.features = self.X.columns
        self.cuisines = self.clf.classes_
        self.cuisine_pca = self._cuisine_pca()
        self.cuisine_distances = self._calculate_cuisine_distances()

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

        # Remove recipes that have no ingredients after filtering
        valid_recipes = X.sum(axis=1) > 0
        X = X[valid_recipes]
        y = y[valid_recipes]
        return X, y

    def _train_classifier(self):
        clf = LinearSVC(C=.1)
        clf.fit(self.X, self.y)
        return clf

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
        df["predicted_cuisine"] = self.clf.predict(recipe_ingredient_matrix)
        return df

    def _calculate_cuisine_distances(self):
        """Calculate pairwise distances between cuisines based on PCA vectors."""
        from scipy.spatial.distance import pdist, squareform

        # Calculate pairwise Euclidean distances between cuisine PCA vectors
        distances = pdist(self.cuisine_pca, metric="euclidean")

        # Convert to square matrix and wrap in DataFrame with cuisine labels
        distance_matrix = squareform(distances)
        return pd.DataFrame(
            distance_matrix,
            index=self.cuisine_pca.index,
            columns=self.cuisine_pca.index,
        )

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

    def avg_dist_from_other_cuisines(self, cuisines_so_far):
        if not cuisines_so_far:
            raise ValueError("The input list of cuisines cannot be empty.")
        distances = self.cuisine_distances[list(cuisines_so_far)]
        distances = distances.drop(cuisines_so_far)
        average_distances = distances.mean(axis=1)
        return average_distances.sort_values()
