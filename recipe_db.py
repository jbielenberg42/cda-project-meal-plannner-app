import pandas as pd
import json


class RecipeDB:

    def __init__(self, cuisine_classifier):
        self.cuisine_classifier = cuisine_classifier
        self.recipes = self.load_and_clean_data()
        # Extract features for all recipes
        self.recipes['features'] = self.cuisine_classifier.extract_features_from_new_recipes(
            self.recipes['ingredients'].apply(lambda x: x.split())
        )
        # Predict cuisines for all recipes
        self.recipes['predicted_cuisine'] = self.cuisine_classifier.clf.predict(
            pd.concat(self.recipes['features'].tolist()).reset_index(drop=True)
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
