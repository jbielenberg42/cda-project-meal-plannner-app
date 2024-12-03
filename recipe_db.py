import pandas as pd
import json
import numpy as np


class RecipeDB:

    def __init__(self, cuisine_classifier):
        self.cuisine_classifier = cuisine_classifier
        self.recipes = self.load_and_clean_data()
        self.recipe_ingredient_matrix = (
            cuisine_classifier.extract_features_from_new_recipes(
                self.recipes["ingredients"]
            )
        )
        
        # Remove recipes with no features after classifier processing
        valid_recipes = (self.recipe_ingredient_matrix.sum(axis=1) > 0)
        self.recipes = self.recipes[valid_recipes].reset_index(drop=True)
        self.recipe_ingredient_matrix = self.recipe_ingredient_matrix[valid_recipes]
        
        # Print info about removed recipes
        num_removed = (~valid_recipes).sum()
        if num_removed > 0:
            print(f"Removed {num_removed} recipes with no features after classifier processing")
        
        self.recipe_cuisine_predictions = cuisine_classifier.cuisine_prediction_df(
            self.recipe_ingredient_matrix
        )


    def load_and_clean_data(self):
        with open("data/recipes_raw_nosource_epi.json", "r") as json_f:
            epicurious_df = pd.DataFrame(json.load(json_f)).T.reset_index(drop=True)

        with open("data/recipes_raw_nosource_fn.json", "r") as json_f:
            fn_df = pd.DataFrame(json.load(json_f)).T.reset_index(drop=True)

        with open("data/recipes_raw_nosource_ar.json", "r") as json_f:
            ar_df = pd.DataFrame(json.load(json_f)).T.reset_index(drop=True)
        df = pd.concat([epicurious_df, fn_df, ar_df], ignore_index=True)
        df = df[["title", "instructions", "ingredients"]]
        df = df.dropna().reset_index(drop=True)
        df["ingredients"] = df["ingredients"].apply(
            lambda x: " ".join(x) if isinstance(x, list) else x
        )
        return df

